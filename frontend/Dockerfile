# --- Build stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Wichtig: VITE_API_BASE muss vor dem Build gesetzt sein (siehe .env.production)
RUN npm run build

# --- Serve stage ---
FROM nginx:1.27-alpine
# Nginx Config ersetzen (Proxy /api -> backend:8000, SPA fallback)
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Statisches React-Build
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
