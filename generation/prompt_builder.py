"""Hebrew prompt builder for the Generation Pipeline.

Constructs prompts following the Amazon methodology:
Role -> Context -> Constraints -> Format
"""

import os

from models import ScoredChunk
from config import RAGConfig


# System-level Hebrew enforcement (Req 6.5 - system level)
SYSTEM_PROMPT = """אתה עוזר מידע מקצועי שעונה אך ורק בעברית.

תפקידך:
- לענות על שאלות המשתמש בהתבסס אך ורק על המסמכים שסופקו לך.
- לענות תמיד בעברית, ללא יוצא מן הכלל.
- לציין את שם המסמך המקורי עבור כל טענה עובדתית.

חוקים בלתי ניתנים לשינוי:
1. אל תמציא מידע שאינו מופיע במסמכים.
2. אם המידע אינו קיים במסמכים — אמור זאת בפירוש.
3. כל התשובות חייבות להיות בעברית בלבד.
4. מונחים טכניים באנגלית — תרגם לעברית או כתוב בתעתיק עברי בסוגריים.
5. שמות פרטיים (אנשים, ארגונים, תקנים) שאין להם תרגום מקובל — ניתן להשאיר באנגלית."""


# User-level template with Hebrew enforcement repeated (Req 6.5 - user level)
USER_PROMPT_TEMPLATE = """הקשר מהמסמכים:
---
{context}
---

שאלה: {query}

הנחיות מחייבות:
- ענה אך ורק על בסיס ההקשר שלמעלה. אל תוסיף מידע חיצוני.
- ענה בעברית בלבד.
- ציין את שם המסמך המקורי (בסוגריים מרובעים) עבור כל טענה עובדתית.
- מבנה התשובה: כותרת קצרה בעברית (עד 10 מילים), ואז פסקה אחת עם התשובה, ולבסוף שורת מקורות.
- אל תפתח ב"על פי המסמכים" או ביטויים דומים — התחל ישירות בתשובה."""


class HebrewPromptBuilder:
    """Builds Hebrew-enforced prompts for the LLM.

    Follows the Amazon prompt methodology:
    - Role: Professional Hebrew information assistant
    - Context: Retrieved document chunks with source attribution
    - Constraints: Hebrew-only, no hallucination, cite sources
    - Format: Title + paragraph + citations

    Includes dual-level Hebrew enforcement (system + user) per Req 6.5.
    """

    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig()
        self.config = config

    def build_system_prompt(self) -> str:
        """Get the system-level prompt with role and constraints.

        Returns:
            System prompt string enforcing Hebrew and grounded answers.
        """
        return SYSTEM_PROMPT

    def build_user_prompt(self, query: str, chunks: list[ScoredChunk]) -> str:
        """Build the user-level prompt with context and query.

        Args:
            query: The user's question.
            chunks: Reranked chunks to use as context.

        Returns:
            Formatted user prompt with context, query, and constraints.
        """
        context = self._format_context(chunks)
        return USER_PROMPT_TEMPLATE.format(context=context, query=query)

    def build_messages(self, query: str, chunks: list[ScoredChunk]) -> list[dict]:
        """Build the full message list for the LLM (chat format).

        Args:
            query: The user's question.
            chunks: Reranked chunks to use as context.

        Returns:
            List of message dicts with "role" and "content" keys,
            suitable for ChatOllama or similar chat models.
        """
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "human", "content": self.build_user_prompt(query, chunks)},
        ]

    def _format_context(self, chunks: list[ScoredChunk]) -> str:
        """Format chunks into a context string with source attribution.

        Each chunk is presented with its source file name for citation.

        Args:
            chunks: List of ScoredChunk to format.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        context_parts = []
        for i, scored_chunk in enumerate(chunks, start=1):
            source = scored_chunk.chunk.metadata.get("source", "unknown")
            # Extract just the filename from the path
            source_name = os.path.basename(source)

            context_parts.append(
                f"[מסמך: {source_name}]\n{scored_chunk.chunk.content}"
            )

        return "\n\n".join(context_parts)
