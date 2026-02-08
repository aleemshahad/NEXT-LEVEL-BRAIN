RETAIL_WEIGHT = 0.2
SEMI_PROFESSIONAL_WEIGHT = 0.5
PROFESSIONAL_ANALYST_WEIGHT = 0.8
INSTITUTIONAL_MACRO_WEIGHT = 1.0
NOISE_WEIGHT = 0.0

# Sentiment thresholds
EXTREME_OPTIMISM = 0.8
EXTREME_PESSIMISM = -0.8
NEUTRAL_ZONE_LOW = -0.2
NEUTRAL_ZONE_HIGH = 0.2

# Divergence thresholds
HIGH_DIVERGENCE_THRESHOLD = 0.5  # If retail vs Inst difference > 0.5
CONTRARIAN_OPPORTUNITY_SCORE = 0.7

# Risk Limits
MAX_SENTIMENT_EXPOSURE = 1.0
MIN_CONTRARIAN_CONFIDENCE = 0.6

# AI / LLM Configuration
LLM_PROVIDER = "groq"  # Options: "openai", "anthropic", "groq", "local"
LLM_MODEL_NAME = "openai/gpt-oss-120b" # Default model

OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Groq Configuration
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# Data Source APIs
# Twitter / X (Tweepy)
TWITTER_BEARER_TOKEN = "your_twitter_bearer_token"
TWITTER_API_KEY = "your_twitter_api_key"
TWITTER_API_SECRET = "your_twitter_api_secret"

# Reddit (PRAW)
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
REDDIT_USER_AGENT = "market_intelligence_bot/1.0"
