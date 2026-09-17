import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_OUTPUT_DIR = "generated_website"


class WebsiteGenerator:
    """Generate a portfolio-safe static HTML/CSS website from confirmed requirements."""

    def __init__(self, llm: Any, retriever: Optional[Any] = None, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.llm = llm
        self.retriever = retriever
        self.output_dir = Path(output_dir)

    def _requirements_text(self, requirements: Dict[str, str]) -> str:
        labels = {
            "product_name": "Product / Project Name",
            "project_type": "Project Type",
            "purpose": "Purpose",
            "target_audience": "Target Audience",
            "design_style": "Design Style",
            "core_features": "Core Features",
            "content_needs": "Content Needs",
            "technical_requirements": "Technical Requirements",
            "special_requests": "Special Requests",
        }

        lines = []
        for key, label in labels.items():
            value = str(requirements.get(key, "")).strip()
            if value:
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def _retrieve_design_context(self, requirements: Dict[str, str]) -> str:
        if self.retriever is None:
            return "No external design knowledge retrieved."

        query_parts = [
            "website design layout navigation content visual style",
            requirements.get("project_type", ""),
            requirements.get("design_style", ""),
            requirements.get("core_features", ""),
        ]
        query = " | ".join(part.strip() for part in query_parts if part.strip())

        try:
            documents = self.retriever.invoke(query)
            chunks = [
                doc.page_content.strip()
                for doc in documents or []
                if getattr(doc, "page_content", "").strip()
            ]
            if not chunks:
                return "No relevant design knowledge retrieved."
            logging.info(f"Retrieved {len(chunks)} design-context chunks for website generation.")
            return "\n\n---\n\n".join(chunks[:4])
        except Exception as exc:
            logging.warning(f"Website-generation retrieval failed: {exc}")
            return "Design knowledge retrieval was unavailable."

    def _build_prompt(self, requirements: Dict[str, str], design_context: str) -> str:
        return f"""
You are a senior front-end developer generating a complete static website.

Use ONLY the CONFIRMED USER REQUIREMENTS below.
Do not invent products, services, pages, features, prices, users, reviews,
statistics, contact details, addresses, social links, or business claims.

If a requirement says the user has no idea, do not invent a feature for it.
You may create neutral visual structure needed to make the page usable,
but it must not introduce new business functionality.

CONFIRMED USER REQUIREMENTS:
{self._requirements_text(requirements)}

RELEVANT DESIGN KNOWLEDGE:
{design_context}

The knowledge above is reference material only. Never copy an example project
from it and never treat its content as this user's requirement.

OUTPUT FORMAT:
Return exactly two sections using these markers:
===HTML===
[complete index.html]
===CSS===
[complete styles.css]
===END===

HTML REQUIREMENTS:
- Complete HTML5 document.
- Use semantic elements such as header, nav, main, section and footer when appropriate.
- Link the stylesheet as ./styles.css.
- Use the confirmed project name and confirmed content only.
- If images are requested, use local placeholder paths such as assets/book-1.jpg;
  do not invent remote image URLs.
- Do not include JavaScript.
- Do not include frameworks or external CDNs.

CSS REQUIREMENTS:
- Complete responsive stylesheet.
- Match the confirmed design style and visual preferences.
- Include mobile responsiveness.
- Use clean layout, spacing, typography, buttons/cards where supported by requirements.
- Do not add unsupported functionality.

QUALITY:
- The result must be coherent and ready to open locally in a browser.
- Do not explain your code.
- Return only the two requested code sections.
""".strip()

    @staticmethod
    def _extract_sections(response: str) -> Tuple[str, str]:
        text = str(response or "").strip()
        text = text.replace("```html", "").replace("```css", "").replace("```", "").strip()

        html_match = re.search(r"===HTML===\s*(.*?)\s*===CSS===", text, re.IGNORECASE | re.DOTALL)
        css_match = re.search(r"===CSS===\s*(.*?)\s*===END===", text, re.IGNORECASE | re.DOTALL)

        if not html_match or not css_match:
            raise ValueError("LLM output did not contain the required HTML/CSS sections.")

        html = html_match.group(1).strip()
        css = css_match.group(1).strip()

        if "<html" not in html.lower() or "<body" not in html.lower():
            raise ValueError("Generated HTML is incomplete.")

        if "styles.css" not in html:
            raise ValueError("Generated HTML does not link styles.css.")

        if len(html) < 300 or len(css) < 200:
            raise ValueError("Generated HTML/CSS is too short.")

        return html, css

    @staticmethod
    def _safe_filename(value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
        return value.strip("-") or "website"

    def _fallback(self, requirements: Dict[str, str]) -> Tuple[str, str]:
        name = requirements.get("product_name", "My Website").strip() or "My Website"
        purpose = requirements.get("purpose", "").strip()
        audience = requirements.get("target_audience", "").strip()
        style = requirements.get("design_style", "").strip()
        content = requirements.get("content_needs", "").strip()
        features = requirements.get("core_features", "").strip()

        hero_text = purpose or "A website designed around the user's requirements."
        audience_text = audience or "Visitors and users of this website."
        content_text = content or "Content will be defined according to the project requirements."
        feature_text = features if features and features.lower() not in {"i have no idea about that", "no idea", "none"} else "Features to be defined."

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{purpose}">
  <title>{name}</title>
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <header class="site-header">
    <div class="container nav-wrap">
      <a class="brand" href="#">{name}</a>
      <nav aria-label="Main navigation">
        <a href="#about">About</a>
        <a href="#content">Content</a>
        <a href="#features">Features</a>
      </nav>
    </div>
  </header>

  <main>
    <section class="hero">
      <div class="container hero-content">
        <p class="eyebrow">{name}</p>
        <h1>{name}</h1>
        <p class="hero-text">{hero_text}</p>
        <a class="button" href="#content">Explore</a>
      </div>
    </section>

    <section id="about" class="section">
      <div class="container two-column">
        <div>
          <p class="eyebrow">Project</p>
          <h2>Built for the intended audience</h2>
        </div>
        <p>{audience_text}</p>
      </div>
    </section>

    <section id="content" class="section section-alt">
      <div class="container">
        <p class="eyebrow">Content</p>
        <h2>What the website presents</h2>
        <p class="lead">{content_text}</p>
      </div>
    </section>

    <section id="features" class="section">
      <div class="container">
        <p class="eyebrow">Features</p>
        <h2>Project direction</h2>
        <div class="feature-card">
          <p>{feature_text}</p>
        </div>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <div class="container">
      <p>{name}</p>
    </div>
  </footer>
</body>
</html>'''

        css = f'''* {{ box-sizing: border-box; }}

:root {{
  --bg: #ffffff;
  --surface: #f5f5f5;
  --text: #171717;
  --muted: #666666;
  --border: #dddddd;
  --accent: #171717;
}}

html {{ scroll-behavior: smooth; }}

body {{
  margin: 0;
  font-family: Inter, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}

a {{ color: inherit; text-decoration: none; }}

.container {{ width: min(1100px, 90%); margin: 0 auto; }}

.site-header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255,255,255,.94);
  border-bottom: 1px solid var(--border);
}}

.nav-wrap {{ min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }}
.brand {{ font-weight: 800; letter-spacing: .04em; }}
nav {{ display: flex; gap: 22px; font-size: .95rem; }}
nav a:hover {{ text-decoration: underline; }}

.hero {{ padding: 120px 0; background: linear-gradient(135deg, #ffffff, #eeeeee); }}
.hero-content {{ max-width: 760px; }}
.eyebrow {{ margin: 0 0 10px; text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; font-weight: 700; color: var(--muted); }}
h1 {{ margin: 0 0 20px; font-size: clamp(3rem, 8vw, 6rem); line-height: .95; letter-spacing: -.05em; }}
h2 {{ margin: 0 0 18px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.05; letter-spacing: -.03em; }}
.hero-text, .lead {{ max-width: 700px; color: var(--muted); font-size: 1.15rem; }}
.button {{ display: inline-block; margin-top: 18px; padding: 12px 22px; background: var(--accent); color: #fff; border-radius: 999px; }}

.section {{ padding: 90px 0; }}
.section-alt {{ background: var(--surface); }}
.two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 60px; align-items: start; }}
.feature-card {{ padding: 28px; border: 1px solid var(--border); border-radius: 18px; background: var(--bg); }}
.site-footer {{ padding: 28px 0; border-top: 1px solid var(--border); color: var(--muted); }}

@media (max-width: 700px) {{
  .nav-wrap {{ align-items: flex-start; padding: 18px 0; flex-direction: column; }}
  nav {{ flex-wrap: wrap; }}
  .hero {{ padding: 90px 0; }}
  .two-column {{ grid-template-columns: 1fr; gap: 24px; }}
}}
'''
        return html, css

    def generate(self, requirements: Dict[str, str]) -> Dict[str, str]:
        """Generate and save index.html and styles.css."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        design_context = self._retrieve_design_context(requirements)

        try:
            prompt = self._build_prompt(requirements, design_context)
            response = self.llm.invoke(prompt)
            html, css = self._extract_sections(response)
            source = "llm"
        except Exception as exc:
            logging.warning(f"LLM website generation failed: {exc}")
            logging.info("Using deterministic fallback website template.")
            html, css = self._fallback(requirements)
            source = "fallback"

        html_path = self.output_dir / "index.html"
        css_path = self.output_dir / "styles.css"
        html_path.write_text(html, encoding="utf-8")
        css_path.write_text(css, encoding="utf-8")

        project_name = requirements.get("product_name", "website")
        slug = self._safe_filename(project_name)
        metadata_path = self.output_dir / "generation_info.txt"
        metadata_path.write_text(
            "RAG Conversational AI - Website Generation\n"
            "=" * 50 + "\n\n"
            f"Project: {project_name}\n"
            f"Generation source: {source}\n"
            "Output: index.html + styles.css\n",
            encoding="utf-8"
        )

        logging.info(f"Website generated using {source}: {html_path} and {css_path}")

        return {
            "html": str(html_path),
            "css": str(css_path),
            "info": str(metadata_path),
            "source": source,
            "slug": slug,
        }
