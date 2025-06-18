#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required files exist
if [[ ! -f "docker-compose.production.yml" ]]; then
    echo -e "${RED}❌ docker-compose.production.yml not found${NC}"
    exit 1
fi

# Prompt for domain configuration
echo -e "${YELLOW}📝 Domain Configuration${NC}"
read -p "Enter your domain name (e.g., example.com): " DOMAIN
read -p "Enter your email for Let's Encrypt: " EMAIL

if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
    echo -e "${RED}❌ Domain and email are required${NC}"
    exit 1
fi

# Update docker-compose.yml with actual domain
echo -e "${GREEN}🔧 Configuring domain settings...${NC}"
sed -i.bak "s/yourdomain.com/$DOMAIN/g" docker-compose.production.yml
sed -i.bak "s/your-email@example.com/$EMAIL/g" docker-compose.production.yml

# Update nginx configuration with actual domain
sed -i.bak "s/yourdomain.com/$DOMAIN/g" nginx/conf.d/default.conf
sed -i.bak "s/yourdomain.com/$DOMAIN/g" nginx/conf.d/default.conf.http-only

# Create environment files if they don't exist
echo -e "${GREEN}📄 Setting up environment files...${NC}"

if [[ ! -f "server/.env.production" ]]; then
    echo "Creating server/.env.production from template..."
    cp server/env.production.template server/.env.production
    sed -i.bak "s/yourdomain.com/$DOMAIN/g" server/.env.production
    echo -e "${YELLOW}⚠️  Please edit server/.env.production with your actual configuration${NC}"
fi

if [[ ! -f "next/.env.production" ]]; then
    echo "Creating next/.env.production from template..."
    cp next/env.production.template next/.env.production
    sed -i.bak "s/yourdomain.com/$DOMAIN/g" next/.env.production
fi

# Create directories for SSL certificates
echo -e "${GREEN}📁 Creating SSL certificate directories...${NC}"
mkdir -p certbot/conf
mkdir -p certbot/www

# Initial SSL certificate setup
echo -e "${GREEN}🔒 Setting up SSL certificates...${NC}"
echo "Starting nginx with HTTP-only configuration for certificate validation..."

# Switch to HTTP-only config for certificate validation
cp nginx/conf.d/default.conf.http-only nginx/conf.d/default.conf

# Start nginx with HTTP-only config
docker compose -f docker-compose.production.yml up -d nginx

# Wait for nginx to be ready
sleep 10

# Check if SSL certificates already exist
if [[ -f "certbot/conf/live/$DOMAIN/fullchain.pem" ]]; then
    echo -e "${GREEN}✅ SSL certificates already exist for $DOMAIN${NC}"
    echo "Skipping certificate generation..."
else
    # Request SSL certificate
    echo "Requesting SSL certificate for $DOMAIN and api.$DOMAIN..."
    if ! docker run --rm -it \
        -v ./certbot/conf:/etc/letsencrypt \
        -v ./certbot/www:/var/www/certbot \
        certbot/certbot:latest \
        certonly --webroot --webroot-path=/var/www/certbot/ \
        --email $EMAIL --agree-tos --no-eff-email \
        -d $DOMAIN -d api.$DOMAIN; then
        echo -e "${RED}❌ SSL certificate generation failed!${NC}"
        echo "This might be due to:"
        echo "1. DNS not pointing to this server"
        echo "2. Port 80 not accessible from the internet"
        echo "3. Domain validation failed"
        echo ""
        echo "You can:"
        echo "1. Fix the DNS/firewall issues and run the script again"
        echo "2. Use existing certificates if you have them"
        echo "3. Continue without SSL (not recommended for production)"
        echo ""
        read -p "Do you want to continue without SSL? (y/N): " CONTINUE_WITHOUT_SSL
        if [[ "$CONTINUE_WITHOUT_SSL" != "y" && "$CONTINUE_WITHOUT_SSL" != "Y" ]]; then
            echo "Aborting deployment. Please fix the issues and try again."
            exit 1
        else
            echo -e "${YELLOW}⚠️  Continuing without SSL certificates...${NC}"
            USE_SSL=false
        fi
    else
        echo -e "${GREEN}✅ SSL certificates generated successfully!${NC}"
        USE_SSL=true
    fi
fi

# Verify SSL certificates exist (unless we're skipping SSL)
if [[ "${USE_SSL:-true}" == "true" ]]; then
    if [[ ! -f "certbot/conf/live/$DOMAIN/fullchain.pem" ]]; then
        echo -e "${RED}❌ SSL certificates not found after generation!${NC}"
        exit 1
    fi
fi

# Create the complete configuration (with or without SSL)
if [[ "${USE_SSL:-true}" == "true" ]]; then
    echo -e "${GREEN}🔧 Creating SSL configuration...${NC}"
    cat > nginx/conf.d/default.conf << EOF
# Rate limiting
limit_req_zone \$binary_remote_addr zone=general:10m rate=10r/s;

# HTTP server - for Let's Encrypt validation and redirects
server {
    listen 80;
    server_name $DOMAIN api.$DOMAIN;

    # Let's Encrypt challenges
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
}

# HTTPS server for frontend
server {
    listen 443 ssl;
    http2 on;
    server_name $DOMAIN;

    # SSL configuration - certificates are mounted at /etc/nginx/ssl/
    ssl_certificate /etc/nginx/ssl/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/live/$DOMAIN/privkey.pem;
    
    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req zone=general burst=20 nodelay;

    location / {
        proxy_pass http://frontend:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# HTTPS server for API
server {
    listen 443 ssl;
    http2 on;
    server_name api.$DOMAIN;

    # SSL configuration - certificates are mounted at /etc/nginx/ssl/
    ssl_certificate /etc/nginx/ssl/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/live/$DOMAIN/privkey.pem;
    
    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req zone=general burst=20 nodelay;

    # File upload size
    client_max_body_size 100M;

    location / {
        proxy_pass http://server:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Longer timeouts for video processing
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF
else
    echo -e "${YELLOW}🔧 Creating HTTP-only configuration...${NC}"
    cat > nginx/conf.d/default.conf << EOF
# Rate limiting
limit_req_zone \$binary_remote_addr zone=general:10m rate=10r/s;

# HTTP server
server {
    listen 80;
    server_name $DOMAIN api.$DOMAIN;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Rate limiting
    limit_req zone=general burst=20 nodelay;

    # Frontend
    location / {
        if (\$host = api.$DOMAIN) {
            proxy_pass http://server:3001;
        }
        if (\$host = $DOMAIN) {
            proxy_pass http://frontend:3000;
        }
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # File upload size for API
        client_max_body_size 100M;
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF
fi

# Verify the config was created properly
echo "Verifying configuration was created..."
if [[ "${USE_SSL:-true}" == "true" ]]; then
    if ! grep -q "listen 443 ssl" nginx/conf.d/default.conf; then
        echo -e "${RED}❌ SSL configuration not properly created!${NC}"
        echo "Current config:"
        head -20 nginx/conf.d/default.conf
        exit 1
    fi
else
    if ! grep -q "listen 80" nginx/conf.d/default.conf; then
        echo -e "${RED}❌ HTTP configuration not properly created!${NC}"
        echo "Current config:"
        head -20 nginx/conf.d/default.conf
        exit 1
    fi
fi

# Test nginx configuration before reload
echo "Testing nginx configuration..."
if ! docker compose -f docker-compose.production.yml exec nginx nginx -t; then
    echo -e "${RED}❌ Nginx configuration test failed!${NC}"
    exit 1
fi

# Reload nginx with new SSL configuration
echo "Reloading nginx with SSL configuration..."
if ! docker compose -f docker-compose.production.yml exec nginx nginx -s reload; then
    echo -e "${RED}❌ Nginx reload failed!${NC}"
    exit 1
fi

# Verify nginx is listening on the correct ports
if [[ "${USE_SSL:-true}" == "true" ]]; then
    echo "Verifying HTTPS is working..."
    sleep 3
    if ! docker compose -f docker-compose.production.yml exec nginx netstat -tlnp | grep -q ":443 "; then
        echo -e "${RED}❌ Nginx is not listening on port 443!${NC}"
        docker compose -f docker-compose.production.yml exec nginx netstat -tlnp
        exit 1
    fi

    # Test HTTPS access
    echo "Testing HTTPS access..."
    sleep 2
    if curl -I https://$DOMAIN > /dev/null 2>&1; then
        echo -e "${GREEN}✅ HTTPS is working!${NC}"
    else
        echo -e "${YELLOW}⚠️  HTTPS test failed, but this might be due to DNS propagation${NC}"
    fi
else
    echo "Verifying HTTP is working..."
    sleep 3
    if ! docker compose -f docker-compose.production.yml exec nginx netstat -tlnp | grep -q ":80 "; then
        echo -e "${RED}❌ Nginx is not listening on port 80!${NC}"
        docker compose -f docker-compose.production.yml exec nginx netstat -tlnp
        exit 1
    fi

    # Test HTTP access
    echo "Testing HTTP access..."
    sleep 2
    if curl -I http://$DOMAIN > /dev/null 2>&1; then
        echo -e "${GREEN}✅ HTTP is working!${NC}"
    else
        echo -e "${YELLOW}⚠️  HTTP test failed, but this might be due to DNS propagation${NC}"
    fi
fi

# Start remaining services if not already running
echo -e "${GREEN}🚀 Starting all services...${NC}"
docker compose -f docker-compose.production.yml up -d

# Verify services are running
echo -e "${GREEN}🔍 Checking service status...${NC}"
sleep 5
docker compose -f docker-compose.production.yml ps

# Setup certificate renewal cron job
echo -e "${GREEN}⏰ Setting up certificate auto-renewal...${NC}"
cat > renew-certs.sh << 'EOF'
#!/bin/bash
docker run --rm -it \
    -v ./certbot/conf:/etc/letsencrypt \
    -v ./certbot/www:/var/www/certbot \
    certbot/certbot:latest \
    renew --webroot --webroot-path=/var/www/certbot/

docker compose -f docker-compose.production.yml exec nginx nginx -s reload
EOF

chmod +x renew-certs.sh

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Configure your DNS to point $DOMAIN and api.$DOMAIN to this server"
echo "2. Edit server/.env.production with your API keys and configuration"
echo "3. Add the following cron job for certificate renewal:"
echo "   0 3 * * * cd $(pwd) && ./renew-certs.sh"
echo
echo -e "${GREEN}🌐 Your application should be available at:${NC}"
if [[ "${USE_SSL:-true}" == "true" ]]; then
    echo "   Frontend: https://$DOMAIN"
    echo "   API: https://api.$DOMAIN"
else
    echo "   Frontend: http://$DOMAIN"
    echo "   API: http://api.$DOMAIN"
    echo -e "${YELLOW}⚠️  Note: Running without SSL. Consider setting up SSL certificates for production.${NC}"
fi
echo
echo -e "${YELLOW}🔍 To check status:${NC}"
echo "   docker compose -f docker-compose.production.yml ps"
echo "   docker compose -f docker-compose.production.yml logs -f" 
