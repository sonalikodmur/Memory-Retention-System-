import random
import uuid
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)
FACTS = [
    "User's favorite color is blue",
    "User lives in Solapur",
    "User's birthday is in March",
    "User owns a dog named Max",
    "User studied at NK Orchid College",
    "User is allergic to peanuts",
    "User has two younger siblings",
    "User's phone number ends in 4471",
    "User works part-time at a bakery",
    "User is left-handed",
    "User's mother tongue is Marathi",
    "User wears glasses",
    "User has a cat named Whiskers",
    "User's hometown is Pune",
    "User is vegetarian",
]

PREFERENCES = [
    "User prefers dark mode",
    "User likes concise answers",
    "User prefers Python over Java",
    "User dislikes long meetings",
    "User prefers email over calls",
    "User prefers tea over coffee",
    "User likes minimalist design",
    "User prefers working late at night",
    "User dislikes spicy food",
    "User prefers written instructions over videos",
    "User likes detailed explanations",
    "User prefers Android over iOS",
]

GOALS = [
    "User wants to learn Spanish",
    "User is learning Spanish",
    "User wants to get better at DSA",
    "User is preparing for interviews",
    "User wants to publish a research paper",
    "User wants to run a marathon next year",
    "User plans to run a marathon next year",
    "User wants to switch careers into AI",
    "User wants to buy a house within five years",
    "User is saving up to buy a house",
    "User wants to learn guitar",
    "User wants to lose 5 kg",
]

TASKS = [
    "Pay electricity bill by Friday",
    "Submit assignment by Monday",
    "Renew driving license this month",
    "Book flight tickets before price increase",
    "Call the dentist to reschedule appointment",
    "Finish reading the assigned book by next week",
    "Send the report to manager by end of day",
    "Return the library books by Thursday",
]

EVENTS = [
    "User watched a movie yesterday",
    "User attended a wedding last weekend",
    "User went on a trip to Goa",
    "User had a job interview last week",
    "User celebrated their birthday recently",
    "User attended a college hackathon",
    "User visited the doctor last month",
]

MEMORY_TYPES = {
    "fact": FACTS,
    "preference": PREFERENCES,
    "goal": GOALS,
    "task": TASKS,
    "event": EVENTS,
}

CONFLICT_PAIRS = [
    ("User prefers tea over coffee", "User prefers coffee over tea"),
    ("User prefers dark mode", "User prefers light mode"),
    ("User dislikes spicy food", "User loves spicy food"),
    ("User prefers Python over Java", "User prefers Java over Python"),
    ("User is vegetarian", "User eats non-vegetarian food regularly"),
]

EVOLUTION_PAIRS = [
    ("User is learning Spanish", "User is now fluent in Spanish"),
    ("User wants to buy a house within five years", "User bought a house"),
    ("User is preparing for interviews", "User got a new job after interviews"),
    ("User wants to run a marathon next year", "User completed a marathon"),
    ("User is saving up to buy a house", "User bought a house"),
]

def generate_memory(now):
    mem_type = random.choice(list(MEMORY_TYPES.keys()))
    text = random.choice(MEMORY_TYPES[mem_type])

    created_days_ago = random.randint(1, 180)
    created_at = now - timedelta(days=created_days_ago)

    last_accessed_days_ago = random.randint(0, created_days_ago)
    last_accessed_at = now - timedelta(days=last_accessed_days_ago)

    access_count = random.choices(
        [0, 1, random.randint(2, 5), random.randint(6, 30)],
        weights=[0.2, 0.3, 0.3, 0.2]
    )[0]

    feedback = random.choices(
        ["positive", "negative", "none"],
        weights=[0.2, 0.1, 0.7]
    )[0]

    return {
        "memory_id": str(uuid.uuid4())[:8],
        "text": text,
        "type": mem_type,
        "created_at": created_at,
        "last_accessed_at": last_accessed_at,
        "access_count": access_count,
        "feedback": feedback,
    }

def generate_conflict_and_evolution_memories(now):
    special_memories = []

    for text_a, text_b in CONFLICT_PAIRS:
        created_a = now - timedelta(days=random.randint(30, 90))
        created_b = created_a + timedelta(days=random.randint(1, 20))
        for text, created in [(text_a, created_a), (text_b, created_b)]:
            special_memories.append({
                "memory_id": str(uuid.uuid4())[:8],
                "text": text,
                "type": "preference" if "prefer" in text.lower() or "dislike" in text.lower() or "love" in text.lower() else "fact",
                "created_at": created,
                "last_accessed_at": created + timedelta(days=random.randint(0, 5)),
                "access_count": random.randint(0, 5),
                "feedback": random.choice(["positive", "none", "negative"]),
            })

    for text_old, text_new in EVOLUTION_PAIRS:
        created_old = now - timedelta(days=random.randint(60, 120))
        created_new = created_old + timedelta(days=random.randint(20, 50))
        for text, created, mem_type in [(text_old, created_old, "goal"), (text_new, created_new, "fact")]:
            special_memories.append({
                "memory_id": str(uuid.uuid4())[:8],
                "text": text,
                "type": mem_type,
                "created_at": created,
                "last_accessed_at": created + timedelta(days=random.randint(0, 5)),
                "access_count": random.randint(0, 5),
                "feedback": random.choice(["positive", "none", "negative"]),
            })

    return pd.DataFrame(special_memories)


def generate_dataset(n=500):
    now = datetime(2026, 7, 29)
    memories = [generate_memory(now) for _ in range(n)]
    df = pd.DataFrame(memories)
    return df

    
if __name__ == "__main__":
    now = datetime(2026, 7, 29)
    df = generate_dataset(500)
    special_df = generate_conflict_and_evolution_memories(now)
    df = pd.concat([df, special_df], ignore_index=True)

    df.to_csv("data/raw/synthetic_memories.csv", index=False)
    print(f"Generated {len(df)} memories total ({len(special_df)} deliberate conflict/evolution pairs).")
    print(special_df[["text", "type", "created_at"]])