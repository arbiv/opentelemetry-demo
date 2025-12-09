# Observable Agent Tutorial

This tutorial will guide you through setting up and using the Observable AI
Agent Tutorial, based on the OpenTelemetry Demo, including OpenTelemetry instrumentation,
Langfuse integration, and LLM judge scoring.

⭐ **Start this repo to show your love for observable AI agents!**

## Prerequisites

- Docker and Docker Compose installed
- Git installed
- OpenAI API key for using real LLM models

## Step 1: Running the Demo and Viewing Traces in Jaeger

### 1.1 Start the Demo

Add these lines to .env.override to disable auto load generation
```
# Disable load generator autostart
LOCUST_AUTOSTART=false
```

From the root directory of the repository, start the demo:

```bash
make start
```

This will start all services including:

- The Astronomy Shop frontend
- All microservices
- Jaeger for distributed tracing
- Grafana for metrics
- OpenTelemetry Collector

### 1.2 Access the Demo

Once the services are running (this may take a few minutes), access:

- **Webstore**: [http://localhost:8080](http://localhost:8080)
- **Jaeger UI**: [http://localhost:8080/jaeger/ui](http://localhost:8080/jaeger/ui)

### 1.3 View Traces in Jaeger

1. Open Jaeger UI at `http://localhost:8080/jaeger/ui`
2. In the Jaeger interface:
   - Select a service from the dropdown (e.g., `frontend`, `product-reviews`)
   - Click "Find Traces"
3. Browse through the traces to see:
   - Service dependencies
   - Request flow across microservices
   - Span durations and attributes
   - Error traces (if any)

4. Click on any trace to see detailed information:
   - Timeline view showing span relationships
   - Tags and attributes for each span
   - Logs associated with spans

### 1.4 Generate Some Traffic

To see more traces, interact with the webstore:

- Browse products
- View product details

Each interaction will generate traces that you can view in Jaeger.

---

## Step 2: Adding OpenAI Keys and Using the "ASK AI" Feature

### 2.1 Configure OpenAI API Keys

To run our AI agent, we need to configure a real LLM. 
If you don't want or can't use a OpenAI API, you can use the original OpenTelemetry demo with a mock LLM.

**Need an OpenAI API key?** Learn how to create one in the [OpenAI API Keys documentation](https://platform.openai.com/api-keys).

Create or edit the `.env.override` file in the root directory:

```bash
# Add these lines to .env.override
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Note**: Replace `sk-your-openai-api-key-here` with your actual OpenAI API key.

### 2.2 Restart the Product Reviews Service

After adding the API keys, restart the product-reviews service to pick up the
new configuration:

```bash
make restart service=product-reviews
```

### 2.3 Use the "Ask AI" Feature

1. Open the webstore at `http://localhost:8080`
2. Navigate to any product page
3. Look for the "Ask AI About This Product" feature
4. Click on one of the conversation starters and wait for the response

### 2.4 View Traces in Jaeger for AI Requests

1. Go back to Jaeger UI at `http://localhost:8080/jaeger/ui`
2. Select `product-reviews` from the service dropdown
3. Look for traces with the operation name `agent.task`
4. Click on a trace to see:
   - The full request flow
   - LLM API calls
   - Tool invocations (fetching product reviews, product info)
   - Response times for each component
   - Attributes including:
     - `app.product.id`: The product ID
     - `app.product.question`: Your question
     - `app.model`: The model used
     - `app.implementation`: Should show "langchain"

5. Expand the spans to see detailed information about:
   - LLM token usage
   - Tool calls made by the agent
   - Response content length

---

## Step 3: Initialize Langfuse and View Traces

### 3.1 Access Langfuse UI

Langfuse is already running as part of the demo. Access it at:

[http://localhost:3000](http://localhost:3000)

### 3.2 Create Langfuse Account and Project

1. On first access, you'll see the Langfuse login page
2. Click "Sign up" to create a new account
3. After creating your account, create a new project:
   - Give it a name (e.g., "OpenTelemetry Demo")
   - Click "Create Project"

### 3.3 Generate API Keys

1. Navigate to **Settings** → **API Keys** in Langfuse
2. Click "Create API Key"
3. Copy both:
   - **Public Key** (starts with `pk-lf-...`)
   - **Secret Key** (starts with `sk-lf-...`)

**Important**: Save these keys securely. The secret key will only be shown once.

### 3.4 Add Keys to Environment

Add the Langfuse API keys to your `.env.override` file:

```bash
# Add these lines to .env.override
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=http://langfuse:3000
```

**Note**: Replace the placeholder values with your actual keys from Langfuse.

### 3.5 Restart the Product Reviews Service

Restart the service to pick up the Langfuse configuration:

```bash
make restart service=product-reviews
```

### 3.6 Verify Langfuse Integration

1. Go back to the webstore and use the "Ask AI" feature again
2. Open Langfuse UI at `http://localhost:3000`
3. You should see the **Dashboard** view by default, which shows an overview
   of your project's LLM activity

### 3.7 Explore the Langfuse Dashboard

The Langfuse dashboard is the first thing you see when opening Langfuse and
provides comprehensive analytics and insights into your LLM usage including
cost, token usage, and latency.

### 3.8 Explore Traces in Langfuse

Navigate to **Traces** in the Langfuse interface to see detailed information
about individual LLM interactions:

1. Click on **Traces** in the navigation menu
2. You should see traces appearing for each AI assistant interaction
3. Click on a trace to see detailed information:

- **Trace Overview**: High-level information about the trace
- **Sessions**: Traces are organized by session ID (product ID)
- **Metadata**: Product ID, question, and model used
- **Token Usage**: Input and output tokens for cost tracking
- **Latency**: Response times for each component
- **Tool Calls**: When the agent uses tools like:
  - `fetch_product_reviews_tool`
  - `fetch_product_info_tool`

---

## Step 4: Using LLM as a Judge

The LLM judge feature evaluates the quality of AI assistant responses using
another LLM to score them.

### 4.1 Enable LLM Judge Scoring

Add the following to your `.env.override` file:

```bash
# Enable LLM judge scoring
ENABLE_LLM_JUDGE_SCORING=true

# Optional: Use a different model for judging (defaults to the same model as
# the assistant)
LLM_JUDGE_MODEL=gpt-4o-mini
```

### 4.2 Restart the Product Reviews Service

Restart the service to enable judge scoring:

```bash
make restart service=product-reviews
```

### 4.3 Use the AI Assistant

1. Go to the webstore and use the "Ask AI" feature
2. Ask a question about a product
3. The system will automatically:
   - Generate the AI assistant response
   - Use the judge LLM to evaluate the response
   - Create a score in Langfuse

### 4.4 View Judge Scores in Langfuse

1. Open Langfuse UI at `http://localhost:3000`
2. Navigate to **Traces**
3. Click on a trace from an AI assistant interaction
4. Look for the **Scores** section, which should show:
   - **Score Name**: `llm_judge_score`
   - **Score Value**: A number between 0 and 1 (higher is better)
   - **Comment**: Reasoning from the judge LLM
   - **Metadata**: Detailed scores for:
     - `relevance_score`: How relevant the response is to the question
     - `accuracy_score`: Factual correctness
     - `completeness_score`: Whether sufficient information was provided
     - `clarity_score`: How clear and well-structured the response is

### 4.5 View Judge Scores in Jaeger

1. Open Jaeger UI at `http://localhost:8080/jaeger/ui`
2. Find a trace for an AI assistant interaction
3. Look for the span attribute:
   - `app.llm_judge.score`: The overall score from the judge

### 4.6 Understanding Judge Scores

The LLM judge evaluates responses on four criteria:

1. **Relevance** (0-1): Does the response directly address the user's question?
2. **Accuracy** (0-1): Is the information provided factually correct?
3. **Completeness** (0-1): Does the response provide sufficient information?
4. **Clarity** (0-1): Is the response clear and well-structured?

The **overall_score** is a single value (0-1) representing the overall quality,
and the judge provides reasoning for its evaluation.

---

## Troubleshooting

### Demo Not Starting

- Check Docker is running: `docker ps`
- View logs: `docker compose logs`
- Check for port conflicts (especially ports 8080, 3000)

### OpenAI API Not Working

- Verify your API key is correct in `.env.override`
- Check the product-reviews service logs: `docker compose logs product-reviews`
- Ensure `LLM_BASE_URL` and `OPENAI_API_KEY` are set correctly
- Ensure docker was started with the make command, so `.env.override` is used

### Langfuse Not Showing Traces

- Verify API keys are set: `docker compose exec product-reviews env | grep LANGFUSE`
- Check Langfuse service is running: `docker compose ps langfuse`
- View product-reviews logs for errors: `docker compose logs product-reviews | grep -i langfuse`

### Judge Scores Not Appearing

- Verify `ENABLE_LLM_JUDGE_SCORING=true` is set
- View product-reviews logs: `docker compose logs product-reviews | grep -i judge`

### Jaeger Not Showing Traces

- Verify Jaeger is running: `docker compose ps jaeger`
- Check OpenTelemetry Collector logs: `docker compose logs otel-collector`
- Ensure services are generating traces (try interacting with the webstore)

---

## Next Steps

- Create a new tool: Add item to cart
- Extend the AI agent to allow a chat experience
- When using the LLM as a Judge, there is an LLM call span with an invalid parent. Why? Can you fix it?

---

## Summary

This tutorial covered:

1. ✅ Running the demo and viewing distributed traces in Jaeger
2. ✅ Configuring OpenAI API keys and using the "Ask AI" feature with trace
   visibility
3. ✅ Setting up Langfuse for LLM-specific observability and viewing
   traces
4. ✅ Enabling and using LLM judge scoring to evaluate response quality

You now have a complete observability setup for your AI-powered application,
with both distributed tracing (Jaeger) and LLM-specific monitoring (Langfuse)
working together!

