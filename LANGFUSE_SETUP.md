# Langfuse Integration

This OpenTelemetry demo includes Langfuse, an open-source LLM observability and monitoring platform. Langfuse is integrated with the LangChain agent in the product-reviews service to provide detailed tracing and monitoring of LLM interactions.

## What is Langfuse?

Langfuse is an open-source platform for LLM engineering that provides:
- **Tracing**: Track every LLM call, token usage, and latency
- **Evaluation**: Monitor model performance and quality
- **Prompt Management**: Version and manage prompts
- **Cost Tracking**: Monitor costs across different models and providers
- **Analytics**: Gain insights into usage patterns and model behavior

## Accessing Langfuse

Once the services are running, you can access Langfuse at:
- Direct access: `http://localhost:3000`
- Via Envoy proxy: `http://localhost:8080/langfuse/`

### First Time Setup

1. On first access, you'll need to create an account
2. After creating an account, create a new project
3. Navigate to **Settings** → **API Keys** to generate your keys
4. Copy the **Public Key** and **Secret Key**

## Configuration

### Environment Variables

The following environment variables can be set to configure Langfuse:

```bash
# Required for Langfuse to work
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional - defaults to internal docker service
LANGFUSE_HOST=http://langfuse:3000

# Optional - Langfuse service configuration
LANGFUSE_PORT=3000
LANGFUSE_NEXTAUTH_URL=http://localhost:3000
LANGFUSE_NEXTAUTH_SECRET=secret-for-development-only-do-not-use-in-production
LANGFUSE_SALT=salt-for-development-only-do-not-use-in-production
LANGFUSE_TELEMETRY_ENABLED=true
LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
```

### Setting API Keys

You can set the Langfuse API keys in several ways:

1. **Using .env file** (recommended):
   ```bash
   echo "LANGFUSE_PUBLIC_KEY=pk-lf-..." >> .env
   echo "LANGFUSE_SECRET_KEY=sk-lf-..." >> .env
   ```

2. **Using environment variables directly**:
   ```bash
   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_SECRET_KEY=sk-lf-...
   docker compose up
   ```

3. **Without API keys** (Langfuse UI still accessible but no traces collected):
   If you don't set the keys, the product-reviews service will still work, but LLM traces won't be sent to Langfuse.

## Features

### What Gets Traced

When the LangChain agent is used (when `USE_LANGCHAIN_AGENT=true`), Langfuse will automatically trace:

1. **LLM Calls**: All interactions with the LLM service
2. **Tool Calls**: When the agent uses tools like `fetch_product_reviews_tool` or `fetch_product_info_tool`
3. **Token Usage**: Input and output tokens for cost tracking
4. **Latency**: Response times for each component
5. **Metadata**: Product IDs, questions, and other contextual information

### Session Tracking

Traces are organized by:
- **Session ID**: Set to the product ID being queried
- **User ID**: Set to `product-reviews-service`
- **Metadata**: Includes the product ID, question, and model used

## Verification

To verify Langfuse is working correctly:

1. Access the OpenTelemetry demo frontend
2. Navigate to a product page
3. Try the "Ask AI Assistant" feature
4. Open Langfuse UI at `http://localhost:3000` or `http://localhost:8080/langfuse/`
5. Navigate to **Traces** in Langfuse to see the captured LLM interactions

## Database

Langfuse shares the PostgreSQL database with other services in the demo. It will automatically create its own tables on first startup.

## Troubleshooting

### No traces appearing in Langfuse

1. Verify API keys are set correctly:
   ```bash
   docker compose exec product-reviews env | grep LANGFUSE
   ```

2. Check product-reviews logs:
   ```bash
   docker compose logs product-reviews | grep -i langfuse
   ```

3. Ensure `USE_LANGCHAIN_AGENT=true` is set (default in docker-compose.yml)

4. Verify Langfuse service is running:
   ```bash
   docker compose ps langfuse
   ```

### Langfuse UI not accessible

1. Check if the service is running:
   ```bash
   docker compose ps langfuse
   ```

2. Check logs for errors:
   ```bash
   docker compose logs langfuse
   ```

3. Verify port 3000 is not already in use by another service

## Development

If you're developing locally and want to use an external Langfuse instance (e.g., Langfuse Cloud):

1. Set `LANGFUSE_HOST` to the external URL:
   ```bash
   export LANGFUSE_HOST=https://cloud.langfuse.com
   ```

2. Use your Langfuse Cloud API keys

3. You can disable the local Langfuse service to save resources:
   ```bash
   docker compose up --scale langfuse=0
   ```

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [LangChain Integration](https://langfuse.com/docs/integrations/langchain)

