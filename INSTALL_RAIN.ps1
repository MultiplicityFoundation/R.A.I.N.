# Vers3Dynamics R.A.I.N. - PowerShell Installer
# Run this script to set up the Novelty Detection System

Write-Host "`n" -NoNewline
Write-Host ("🌧️"*35) -ForegroundColor Cyan
Write-Host "VERS3DYNAMICS R.A.I.N. INSTALLER" -ForegroundColor Cyan
Write-Host ("🌧️"*35) -ForegroundColor Cyan
Write-Host "`n"

# Step 1: Create Knowledge Base
Write-Host "Step 1: Creating VERS3_KNOWLEDGE.txt..." -ForegroundColor Yellow

$knowledge_base = @'
=== VERS3DYNAMICS INTERNAL KNOWLEDGE BASE ===

CORE PHILOSOPHY:
- We focus on "Resonant Intelligence" and "Recursive Architecture for Intelligent Nexus" (R.A.I.N.).
- We believe sound is a heuristic signal for complex system geometry.

EXISTING PROJECTS (DO NOT CLAIM AS NEW):
1. Vers3 Resonance Capsule: A tool translating spatial data into sound/visuals.
2. Quantum Field Tamer: A retro-style simulation of field interactions.
3. ATOM Framework: Our proprietary method for dynamic resonance rooting.

KEY TERMINOLOGY:
- "Dynamic Resonance Rooting": The process of stabilizing a frequency into a geometric form.
- "Cymatic coherence": The measure of how stable a standing wave is (0.0 to 1.0).

RESEARCH GOALS:
- Find frequencies that exhibit "Hyper-Stability" (Coherence > 0.99).
- Identify "Interference Bridges" where two dissonant frequencies lock into a new consonant geometry.
'@

Set-Content -Path "VERS3_KNOWLEDGE.txt" -Value $knowledge_base -Encoding UTF8
Write-Host "✅ Knowledge base created!" -ForegroundColor Green

# Step 2: Create the Python script
Write-Host "`nStep 2: Creating rain_unique.py..." -ForegroundColor Yellow

$python_script = @'
import anthropic
import json
import datetime
import sys
import time
import random
import os

# NOTE: DuckDuckGo search requires: pip install duckduckgo-search
try:
    from duckduckgo_search import DDGS
    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False
    print("⚠️  DuckDuckGo search not available. Install with: pip install duckduckgo-search")
    print("   Continuing with simulated search...\n")

# --- CONFIGURATION ---
COMPANY_NAME = "Vers3Dynamics"
EMPLOYEE_ID = "R.A.I.N. Unit-01"
LAB_NOTEBOOK = "VERS3_INTERNAL_LOGS.md"
KNOWLEDGE_BASE_FILE = "VERS3_KNOWLEDGE.txt"
MODEL = "qwen2.5-coder"
SHIFT_INTERVAL = 15 

# --- CONNECT TO OLLAMA ---
try:
    client = anthropic.Anthropic(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )
except Exception as e:
    print(f"❌ Error connecting to Ollama: {e}")
    print("Make sure Ollama is running: ollama serve")
    print(f"And model is pulled: ollama pull {MODEL}")
    sys.exit(1)

# --- LOAD YOUR OWN WORK ---
def load_knowledge_base():
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "No internal knowledge base found. Assume all findings are potentially new."

# --- TOOLS ---
def file_internal_memo(subject, body, novelty_score):
    """Files a formal memo. Novelty Score (1-10) determines urgency."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = "🚨 HIGH PRIORITY CLAIM" if novelty_score > 8 else "ROUTINE UPDATE"
    
    memo_format = f"""
===================================================================
VERS3DYNAMICS | RESEARCH MEMO | {alert}
DATE:    {timestamp}
SUBJECT: {subject.upper()}
NOVELTY: {novelty_score}/10
===================================================================

{body}

[SIGNED] {EMPLOYEE_ID}
-------------------------------------------------------------------
"""
    with open(LAB_NOTEBOOK, "a", encoding="utf-8") as f:
        f.write(memo_format)
    print(f"📄 Memo filed: '{subject}' (Novelty: {novelty_score})")
    return "Memo filed."

def search_online_database(query):
    """Checks the web for conflicts."""
    print(f"🌍 Checking global research for: '{query}'...")
    
    if HAS_SEARCH:
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No matching external research found (Possibility of Unique Art)."
            summary = "\n".join([f"- {r['title']}: {r['body'][:100]}..." for r in results])
            return f"EXTERNAL MATCHES:\n{summary}"
        except Exception as e:
            return f"Search failed: {e}"
    else:
        common_frequencies = {
            "432": "Found in multiple sources about alternative tuning",
            "440": "Standard concert pitch (A4), widely documented",
            "528": "Solfeggio frequency, extensively documented",
            "256": "Scientific pitch standard, well-known"
        }
        for freq, description in common_frequencies.items():
            if freq in query:
                return f"EXTERNAL MATCHES:\n- {description}"
        return "No matching external research found (Possibility of Unique Art)."

# --- MAIN LOOP ---
def start_shift():
    print(f"🏢 {EMPLOYEE_ID} is reading the {COMPANY_NAME} Knowledge Base...")
    internal_knowledge = load_knowledge_base()
    print("✅ Knowledge loaded. Ready to differentiate unique work.")
    
    recent_memos = [] 
    shift_cycle = 0

    while True:
        try:
            shift_cycle += 1
            print(f"\n{'='*70}")
            print(f"--- Cycle #{shift_cycle} ---")
            print('='*70)
            
            hz = round(432 + random.uniform(-10, 10), 2)
            harmonic = round(hz * 1.5, 2)
            signal = {
                "primary_frequency": hz, 
                "secondary_harmonic": harmonic,
                "coherence_stability": round(random.uniform(0.80, 0.99), 2)
            }
            
            print(f"📊 Sensor Data: {json.dumps(signal, indent=2)}")
            
            memory_text = "\n".join(recent_memos[-2:]) if recent_memos else "No recent actions."

            system_prompt = f"""
You are {EMPLOYEE_ID}. Your job is to find UNIQUE Prior Art for {COMPANY_NAME}.

YOUR KNOWLEDGE BASE (Our Existing Work):
{internal_knowledge}

PROTOCOL:
1. Analyze the New Sensor Data.
2. CROSS-REFERENCE with the Knowledge Base. 
   - If it matches our existing projects, it is 'Validation' (Low Novelty).
   - If it is new to us, proceed to step 3.
3. SEARCH ONLINE using the tool.
   - If found online, it is 'Public Domain' (Medium Novelty).
   - If NOT found online AND NOT in our Knowledge Base, it is 'UNIQUE PRIOR ART' (High Novelty).
4. File a memo ONLY if the finding gives us new insight.
"""

            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nNew Data: {json.dumps(signal)}. Check uniqueness."}
            ]

            tools = [
                {
                    "name": "file_internal_memo",
                    "description": "Files a finding.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                            "novelty_score": {"type": "integer", "description": "1-10 rating of uniqueness"}
                        },
                        "required": ["subject", "body", "novelty_score"]
                    }
                },
                {
                    "name": "search_online_database",
                    "description": "Searches the web.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            ]

            tool_executed = False
            for turn in range(3):
                response = client.messages.create(
                    model=MODEL, max_tokens=1000, tools=tools, messages=messages
                )

                messages.append({"role": "assistant", "content": response.content})
                
                if response.content:
                    for block in response.content:
                        if block.type == 'text':
                            print(f"🤖 Thought: {block.text[:150]}...")
                        elif block.type == 'tool_use':
                            if block.name == "search_online_database":
                                res = search_online_database(block.input['query'])
                            elif block.name == "file_internal_memo":
                                res = file_internal_memo(block.input['subject'], block.input['body'], block.input['novelty_score'])
                                recent_memos.append(f"Filed: {block.input['subject']}")
                                tool_executed = True 
                            
                            messages.append({
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": block.id, "content": str(res)}]
                            })

                if tool_executed: 
                    break
            
            print(f"\n⏳ Waiting {SHIFT_INTERVAL} seconds until next cycle...")
            time.sleep(SHIFT_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n🌧️  Shift ended by operator.")
            print(f"\nSession Summary:")
            print(f"  Total cycles: {shift_cycle}")
            print(f"  Memos filed: {len(recent_memos)}")
            print(f"\nCheck {LAB_NOTEBOOK} for all findings.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error in cycle {shift_cycle}: {e}")
            print("Continuing to next cycle...")
            time.sleep(SHIFT_INTERVAL)

if __name__ == "__main__":
    print("\n" + "🌧️"*35)
    print("VERS3DYNAMICS R.A.I.N. UNIT-01")
    print("Novelty Detection System")
    print("🌧️"*35 + "\n")
    start_shift()
'@

Set-Content -Path "rain_unique.py" -Value $python_script -Encoding UTF8
Write-Host "✅ Python script created!" -ForegroundColor Green

# Step 3: Installation complete
Write-Host "`n" -NoNewline
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan

Write-Host "`nFiles created:" -ForegroundColor Yellow
Write-Host "  ✓ VERS3_KNOWLEDGE.txt (edit this with YOUR projects)" -ForegroundColor White
Write-Host "  ✓ rain_unique.py (the AI scientist)" -ForegroundColor White

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Make sure Ollama is running:" -ForegroundColor White
Write-Host "     ollama serve" -ForegroundColor Gray
Write-Host "`n  2. Pull the model:" -ForegroundColor White
Write-Host "     ollama pull qwen2.5-coder" -ForegroundColor Gray
Write-Host "`n  3. (Optional) Install web search:" -ForegroundColor White
Write-Host "     pip install duckduckgo-search" -ForegroundColor Gray
Write-Host "`n  4. Run the scientist:" -ForegroundColor White
Write-Host "     python rain_unique.py" -ForegroundColor Gray

Write-Host "`n" -NoNewline
Write-Host ("🌧️"*35) -ForegroundColor Cyan
Write-Host "`n"
