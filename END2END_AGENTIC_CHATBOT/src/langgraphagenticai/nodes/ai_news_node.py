from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate


class AINewsNode:
    
    def __init__(self, llm):
        """
        Initialize the AINewsNode with API Keys for Tavily and GROQ
        """
        self.tavily = TavilyClient()
        self.llm = llm 
        
        # this is used to capture various steps in this file so that later can be used for steps shown
        self.state = {}
    
    def _safe(self, text: str) -> str:
        """Sanitize text to remove problematic characters"""
        if not isinstance(text, str):
            text = str(text)
        # Replace non-breaking hyphen and drop any other chars Windows can't encode
        return (text.replace("\u2011", "-")
                    .encode("utf-8", errors="ignore")
                    .decode("utf-8"))
    
    def _sanitize_state(self, state: dict):
        """Recursively sanitize all strings in a state dictionary"""
        clean = {}
        for k, v in state.items():
            if isinstance(v, str):
                clean[k] = self._safe(v)
            elif isinstance(v, list):
                clean[k] = [self._sanitize_state(i) if isinstance(i, dict) else self._safe(str(i)) for i in v]
            elif isinstance(v, dict):
                clean[k] = self._sanitize_state(v)
            else:
                clean[k] = v
        return clean
    
    def _sanitize_news_item(self, item: dict) -> dict:
        """Sanitize a single news item dictionary"""
        sanitized = {}
        for key, value in item.items():
            if isinstance(value, str):
                sanitized[key] = self._safe(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_news_item(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_news_item(i) if isinstance(i, dict) 
                    else self._safe(str(i)) if isinstance(i, str)
                    else i
                    for i in value
                ]
            else:
                sanitized[key] = value
        return sanitized
        
    def fetch_news(self, state: dict) -> dict:
        """
        Fetch AI News based on the specified frequency and domain.

        Args:
            state (dict): The state dictionary containing 'frequency'

        Returns:
            dict: updated state with 'news_data' key containing fetched news.
        """
        
        frequency = state['messages'][0].content.lower()
        self.state['frequency'] = frequency
        time_range_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm', 'year': 'y'}
        days_map = {'daily': 1, 'weekly': 7, 'monthly': 30, 'year': 365}
        
        try:
            response = self.tavily.search(
                query="Top Artificial Intelligence (AI) technology news globally the important ones",
                topic="news",
                time_range=time_range_map[frequency],
                include_answer="advanced",
                max_results=15,
                days=days_map[frequency],
                # include_domains=["techcrunch.com", ......]
            )
            
            # CRITICAL FIX: Sanitize news data immediately after fetching
            raw_news_data = response.get('results', [])
            sanitized_news_data = [self._sanitize_news_item(item) for item in raw_news_data]
            
            state['news_data'] = sanitized_news_data
            self.state['news_data'] = sanitized_news_data
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            state['news_data'] = []
            self.state['news_data'] = []
        
        return state
    
    def summarize_news(self, state: dict) -> dict:
        """
        Summarize the fetched news using an LLM.

        Args:
            state (dict): The state dictionary containing 'news_data'.

        Returns:
            dict: Updated state with 'summary' key containing the summarized news.
        """

        # Get news items (list of dicts) previously stored in fetch_news
        news_items = self.state["news_data"]

        # Convert articles into a plain-text block for the LLM
        articles_text = ""
        for item in news_items:
            title = self._safe(item.get("title", ""))
            date = self._safe(item.get("published_date", ""))
            url = self._safe(item.get("url", ""))
            summary = self._safe(item.get("content", ""))
            articles_text += f"Title: {title}\nDate: {date}\nURL: {url}\nContent: {summary}\n\n"

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an expert AI news editor creating a polished markdown digest.\n"
                    "You receive multiple AI/technology news articles as raw text.\n\n"
                    "Your goals:\n"
                    "1) Identify the most important, high-impact news items.\n"
                    "2) Group related stories together.\n"
                    "3) Write clear, engaging summaries that a busy professional can skim quickly.\n\n"
                    "Output requirements (markdown):\n"
                    "- Start with a short title and 1–2 sentence overview of the overall AI news trend.\n"
                    "- Then create sections for each story using this structure:\n"
                    "  ### [Short, punchy headline]\n"
                    "  - Date: YYYY-MM-DD (IST)\n"
                    "  - Source: [Domain](URL)\n"
                    "  - Category: one of [Research, Product Launch, Policy/Regulation, Funding/Business, Ethics/Safety, Other]\n"
                    "  - Summary: 2–4 concise sentences explaining what happened, why it matters, and who is impacted.\n"
                    "  - Key takeaway: 1 sentence with the main implication.\n\n"
                    "Additional rules:\n"
                    "- Sort stories from newest to oldest.\n"
                    "- Combine very similar articles into a single story and mention that multiple sources reported it.\n"
                    "- Use neutral, professional tone (no hype, no clickbait).\n"
                    "- If some articles are low-quality or duplicates, you may ignore them.\n"
                    "- If any information is unclear, state that it is unclear instead of guessing.\n"
                ),
            ),
            (
                "user",
                (
                    "You are given a list of AI/tech news articles with fields Content, URL and Date.\n"
                    "Use only the information inside these articles.\n\n"
                    "Articles:\n{articles}"
                ),
            ),
        ])

        articles_str = "\n\n".join([
            f"Content: {self._safe(item.get('content', ''))}\nURL: {self._safe(item.get('url', ''))}\nDate: {self._safe(item.get('published_date', ''))}"
            for item in news_items
        ])

        try:
            response = self.llm.invoke(prompt_template.format(articles=articles_str))
            
            # Sanitize LLM output
            safe_content = self._safe(response.content)
            
            state["summary"] = safe_content
            self.state["summary"] = safe_content
            
        except Exception as e:
            print(f"Error summarizing news: {e}")
            state["summary"] = "Error generating summary."
            self.state["summary"] = "Error generating summary."

        return self._sanitize_state(self.state)

    def save_result(self, state):
        """Save the summary to a markdown file"""
        
        frequency = self.state['frequency']
        summary = self._safe(self.state['summary'])
        filename = f"./AINews/{frequency}_summary.md"
        
        try:
            # Ensure directory exists
            import os
            os.makedirs("./AINews", exist_ok=True)
            
            # Force UTF-8 encoding
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
                f.write(summary)
            
            self.state['filename'] = filename
            print(f"Successfully saved summary to {filename}")
            
        except Exception as e:
            print(f"Error saving file: {e}")
            self.state['filename'] = None
        
        return self.state