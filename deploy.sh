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

# Initial SSL certificate setup (dry run first)
echo -e "${GREEN}🔒 Setting up SSL certificates...${NC}"
echo "Starting nginx with HTTP-only configuration for certificate validation..."

# Create a backup of the SSL config with a proper name
cp nginx/conf.d/default.conf nginx/conf.d/default.conf.backup
# Switch to HTTP-only config for certificate validation
cp nginx/conf.d/default.conf.http-only nginx/conf.d/default.conf

# Start nginx with HTTP-only config
docker compose -f docker-compose.production.yml up -d nginx

# Wait for nginx to be ready
sleep 10

# Request SSL certificate
echo "Requesting SSL certificate for $DOMAIN and api.$DOMAIN..."
docker run --rm -it \
    -v ./certbot/conf:/etc/letsencrypt \
    -v ./certbot/www:/var/www/certbot \
    certbot/certbot:latest \
    certonly --webroot --webroot-path=/var/www/certbot/ \
    --email $EMAIL --agree-tos --no-eff-email \
    -d $DOMAIN -d api.$DOMAIN

# Restore the full SSL config from backup
echo -e "${GREEN}🔧 Enabling SSL configuration...${NC}"
cp nginx/conf.d/default.conf.backup nginx/conf.d/default.conf

# Reload nginx with new SSL configuration
echo "Reloading nginx with SSL configuration..."
docker compose -f docker-compose.production.yml exec nginx nginx -s reload

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
echo "   Frontend: https://$DOMAIN"
echo "   API: https://api.$DOMAIN"
echo
echo -e "${YELLOW}🔍 To check status:${NC}"
echo "   docker compose -f docker-compose.production.yml ps"
echo "   docker compose -f docker-compose.production.yml logs -f" 
