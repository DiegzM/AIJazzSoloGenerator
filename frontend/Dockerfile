# Base
FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

RUN npm install

# Copy code
COPY . .

# Dev port
EXPOSE 3000

# Start the app
CMD ["npm", "start"]