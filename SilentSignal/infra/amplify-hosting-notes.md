# Amplify Hosting Notes

For hackathon submission you can host the dashboard with Amplify and point it to API Gateway/Lambda or ECS-hosted FastAPI.

Minimal path:
- Build artifact: `apps/dashboard` static files
- API endpoint: set dashboard API base to deployed FastAPI URL
- CORS: include Amplify app domain in `CORS_ORIGINS`

This MVP keeps hosting setup as docs-only to avoid overbuilding before demo validation.
