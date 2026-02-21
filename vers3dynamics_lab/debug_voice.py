"""
Quick TTS test — validates that all agent voices work independently.
Uses the same fresh-engine-per-call pattern as VoiceManager to avoid COM hangs.
"""
import pyttsx3
import time


def discover_voices():
    """Find David and Zira voice IDs using a temp engine."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(f"Found {len(voices)} voices.")

    david = None
    zira = None

    for v in voices:
        print(f" - {v.name} ({v.id})")
        if 'david' in v.name.lower():
            david = v.id
        if 'zira' in v.name.lower():
            zira = v.id

    engine.stop()
    del engine
    return david, zira


def speak_as(agent_name, text, voice_id, rate):
    """Create a fresh engine, speak, destroy — mirrors VoiceManager.speak()."""
    print(f"\n🔊 Testing {agent_name} (rate={rate})...")
    engine = None
    try:
        engine = pyttsx3.init()
        if voice_id:
            engine.setProperty('voice', voice_id)
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()
        print(f"   ✓ {agent_name} done.")
    except Exception as e:
        print(f"   ✗ {agent_name} failed: {e}")
    finally:
        if engine:
            try:
                engine.stop()
            except:
                pass
            del engine


def test_voices():
    print("Initializing TTS engine for voice discovery...")
    david, zira = discover_voices()

    if not david:
        print("\n⚠ David voice not found.")
    if not zira:
        print("\n⚠ Zira voice not found.")

    # Simulate all four agents in meeting order
    speak_as("James",   "This is James speaking. Testing voice output.",   david, 160)
    speak_as("Jasmine", "This is Jasmine speaking. Testing voice output.", zira or david, 185)
    speak_as("Luca",    "This is Luca speaking. Testing voice output.",    david, 195)
    speak_as("Elena",   "This is Elena speaking. Testing voice output.",   zira or david, 210)

    print("\n✅ TTS Test Complete.")


if __name__ == "__main__":
    test_voices()
