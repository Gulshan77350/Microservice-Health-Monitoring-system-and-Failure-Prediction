# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run lint && npm run build

# Serve stage: nginx renders templates/*.template with envsubst at start-up.
FROM nginx:1.27-alpine
ENV NGINX_ENVSUBST_FILTER=^API_KEY$
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 80
