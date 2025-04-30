# book.py: Multi-agentic book writing with Google Gemini API (local version)
import os
import json
import requests
from duckduckgo_search import ddg
import time # Added for potential rate limiting

# Set your Gemini API key here or via environment variable
GEMINI_API_KEY = "AIzaSyDI0n2BXltPHQc0WYe3QU7_r6trRUs_dWU"

START_DATE = "2024-11-01"
END_DATE   = "2025-04-30"

TARGET_WORDS = 3500  # Aproximativ 10 pagini
MAX_ITERATIONS = 5

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    # Added simple retry logic for potential API errors
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120) # Increased timeout
            response.raise_for_status()
            result = response.json()
            # Check for empty or error response from Gemini
            if not result.get("candidates") or not result["candidates"][0].get("content"):
                 print(f"Warning: Gemini returned unexpected response: {result}")
                 return f"Eroare Gemini: Răspuns neașteptat - {result}" # Return error message
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            print(f"Eroare API Gemini (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5) # Wait before retrying
            else:
                return f"Eroare API Gemini: {e}" # Return error after retries
        except Exception as e: # Catch other potential errors like JSON parsing
             print(f"Eroare neașteptată în call_gemini (attempt {attempt+1}): {e}")
             if attempt < 2:
                 time.sleep(5)
             else:
                 return f"Eroare neașteptată: {e}"


def web_search_news(query):
    sites = [
        "site:digi24.ro", "site:realitatea.net",
        "site:romaniatv.net", "site:antena3.ro"
    ]
    # Return dict with url and title
    found_urls = []
    for site in sites:
        q = f"{site} {query} after:{START_DATE} before:{END_DATE}"
        try:
            results = ddg(q, max_results=5) # Reduced max_results per site for speed
            if results:
                for r in results:
                    url = r.get("href") or r.get("link")
                    title = r.get("title", "Fără titlu") # Get title, default if missing
                    if url:
                        found_urls.append({"url": url, "title": title})
        except Exception as e:
            print(f"Eroare la căutare DDG pentru '{q}': {e}")
            continue # Continue with next site if one fails
    # ...existing code...
    # Return list of dicts
    unique_urls = list({item['url']: item for item in found_urls}.values()) # Keep title associated
    return unique_urls


def reasoner(book_title):
    prompt = (
        f"Pentru titlul '{book_title}', împarte cartea în 3-5 capitole relevante. "
        "Răspunde doar cu un array JSON de forma: "
        "[{'titlu': 'Capitol 1', 'descriere': '...'}, ...] fără niciun alt text."
    )
    return call_gemini(prompt)


# Updated researcher to use topic/description and return title/summary
def researcher(topic, description):
    query = f"{topic} {description}"
    print(f"  Căutare pentru: {query}")
    urls_with_titles = web_search_news(query)
    research_results = []
    print(f"  Am găsit {len(urls_with_titles)} URL-uri unice.")
    for item in urls_with_titles[:5]:  # Limit summaries for demo/speed
        url = item['url']
        title = item['title']
        print(f"    Rezumare: {url}")
        prompt = f"Rezumă pe scurt (max 100 cuvinte) conținutul principal al acestui articol de știri: {url}"
        summary = call_gemini(prompt)
        # Basic check if summary seems valid
        if "Eroare" not in summary and len(summary) > 10:
             research_results.append({"url": url, "title": title, "summary": summary})
        else:
             print(f"    Problemă la rezumarea {url}. Rezumat: {summary}")
    print(f"  Am obținut {len(research_results)} rezumate valide.")
    return research_results


# Updated writer to use topic/research_data and generic prompt
def writer(topic, research_data, existing_text=None):
    context_prompt = "\n".join([f"- {item['summary']} (Sursa: {item['url']})" for item in research_data]) # Use \n for join

    if existing_text:
        # Use string concatenation for multi-line prompt
        prompt = (
            f"Extinde și detaliază următorul text despre '{topic}', folosind informațiile suplimentare din surse dacă este necesar. "
            "Adaugă exemple, explicații, context și analizează implicațiile. Nu repeta informația deja prezentă.\n\n"
            "Text existent:\n"
            f"{existing_text}\n\n"
            "Surse suplimentare (folosește doar dacă aduc informații noi):\n"
            f"{context_prompt}"
        )
    else:
        # Use string concatenation for multi-line prompt
        prompt = (
            f"Scrie un text detaliat pentru un capitol de carte despre '{topic}'. Folosește următoarele rezumate din surse:\n"
            f"{context_prompt}\n\n"
            "Structurează textul logic, cu introducere, dezvoltare și concluzii. Fii cât mai informativ și obiectiv."
        )

    return call_gemini(prompt)


# Updated reference_agent to insert markers better and return only text
def reference_agent(text, research_data):
    # Simple approach: Add markers at the end of the text for now
    # A more robust approach would involve finding relevant sentences/paragraphs
    markers = []
    for idx, item in enumerate(research_data, 1):
        markers.append(f"[{idx}]")
    # Use f-string for clarity and standard newline
    text_with_markers = f"{text}\nSurse: {' '.join(markers)}"
    return text_with_markers


def editor(text):
    prompt = (
        "Editează următorul text pentru stil, fluență și continuitate:\n" + text
    )
    return call_gemini(prompt)


def orchestrator(book_title):
    print("Orchestrator: Generez structură carte...")
    structure = reasoner(book_title)
    try:
        # Handle potential markdown code block ```json ... ```
        if isinstance(structure, str) and structure.strip().startswith("```json"):
             structure = structure.strip()[7:-3].strip()
        structure = json.loads(structure)
    except Exception as e:
        print("Eroare la parsarea structurii JSON de la reasoner! Răspunsul a fost:")
        print(structure)
        print("\nAsigură-te că Gemini răspunde strict cu JSON. Poți ajusta promptul reasoner dacă e nevoie.")
        return
    chapters = structure if isinstance(structure, list) else structure.get("capitole", [])
    if not chapters:
         print("Nu s-a putut genera structura capitolelor. Verifică răspunsul reasoner.")
         return

    book_chapters_content = {} # Use dict to store content per chapter topic
    all_references = []
    reference_map = {} # Map URL to index

    # Initial content generation loop
    for chapter in chapters:
        topic = chapter.get("titlu") or chapter.get("topic") or chapter.get("capitol") or str(chapter)
        description = chapter.get("descriere") or chapter.get("description") or ""
        print(f"\n--- Capitol: {topic} ---") # Use \n
        print(f"Searcher: Caut surse pentru '{topic}'...")
        # Pass topic/description to researcher
        research_data = researcher(topic, description)

        # Update global reference list and map
        chapter_refs = []
        for item in research_data:
            if item['url'] not in reference_map:
                 all_references.append(item)
                 reference_map[item['url']] = len(all_references) # Assign index (1-based)
            chapter_refs.append(item) # Keep track of refs for this chapter

        if not chapter_refs:
             print(f"Nu s-au găsit surse valide pentru '{topic}'. Capitolul va fi omis.")
             book_chapters_content[topic] = "Nu s-au găsit surse pentru acest capitol."
             continue

        print(f"Writer: Scriu draft inițial pentru '{topic}'...")
        # Pass topic/research_data to writer
        draft = writer(topic, chapter_refs)

        # Simple edit pass
        print("Editor: Editez draftul inițial...")
        edited_draft = editor(draft)
        book_chapters_content[topic] = edited_draft

    # Iterative extension loop
    current_words = sum(len(content.split()) for content in book_chapters_content.values())
    print(f"\n--- Faza de extindere (Total cuvinte curent: {current_words}) ---") # Use \n

    for iteration in range(MAX_ITERATIONS):
        if current_words >= TARGET_WORDS:
            print(f"Am atins {current_words} cuvinte (>= {TARGET_WORDS}). Finalizare.")
            break
        print(f"\nExtind conținutul, iterația {iteration+1}... (total: {current_words} cuvinte)") # Use \n

        # Extend the shortest chapter first
        shortest_topic = min(book_chapters_content, key=lambda k: len(book_chapters_content[k].split()))
        print(f"Extind cel mai scurt capitol: '{shortest_topic}'")

        # Find references for this chapter again (could be optimized by storing earlier)
        chapter_refs_for_extension = [ref for ref in all_references if ref['url'] in [item['url'] for item in researcher(shortest_topic, "")]] # Re-fetch refs for context

        # Use writer with existing text for extension
        extended_text = writer(shortest_topic, chapter_refs_for_extension, existing_text=book_chapters_content[shortest_topic])

        # Edit the extended text
        print(f"Editor: Editez textul extins pentru '{shortest_topic}'...")
        edited_extended_text = editor(extended_text)

        # Update chapter content and word count
        word_diff = len(edited_extended_text.split()) - len(book_chapters_content[shortest_topic].split())
        book_chapters_content[shortest_topic] = edited_extended_text
        current_words += word_diff
        print(f"Capitolul '{shortest_topic}' extins cu ~{word_diff} cuvinte.")

    else: # If loop finishes without break
        print(f"Atenție: S-a atins numărul maxim de iterații ({MAX_ITERATIONS}) fără a ajunge la {TARGET_WORDS} cuvinte (total: {current_words}).")


    # Final processing and saving
    print("\n--- Generare fișier final ---") # Use \n
    final_book_text = f"TITLU: {book_title}\n\n" # Use \n
    for topic, content in book_chapters_content.items():
        # Add reference markers based on the final reference list
        # This is a simple placeholder - needs robust implementation
        # to find where each reference is relevant in the *final* text.
        # For now, just list the chapter title and content.
        final_book_text += f"# {topic}\n\n{content}\n\n" # Use \n
    final_book_text += "\nREFERINȚE:\n" # Use \n
    for idx, ref in enumerate(all_references, 1):
        # Use title and url for references
        final_book_text += f"[{idx}] {ref['title']} - {ref['url']}\n" # Use \n
    with open("carte_finala.txt", "w", encoding="utf-8") as f:
        f.write(final_book_text)
    print("\nCartea a fost generată în carte_finala.txt!") # Use \n


def main(book_title):
    orchestrator(book_title)

if __name__ == "__main__":
    main("Alegerile prezidențiale din România 2024")
