"""Text chatbot front-end for the hybrid recommender.

Run:  python chat.py
"""
from recommender import HybridRecommender

HELP = """Things you can say:
  like <title>          tell me a movie you enjoyed
  dislike <title>       tell me one you didn't
  more like <title>     find movies similar to one title
  mood <description>    e.g. mood funny family story about friendship
  recommend             suggestions based on everything you've told me
  popular               crowd favourites
  list                  show all titles
  help / quit
"""


def show(recs):
    if not recs:
        print("  I couldn't find anything for that. Try different words.")
    for i, (title, year, score) in enumerate(recs, 1):
        print(f"  {i}. {title} ({year})  score {score:.2f}")


def main():
    rec = HybridRecommender()
    liked, disliked = [], []
    print("Hi! I'm your movie recommender. Type 'help' to see what I can do.")
    print("To start, here are some crowd favourites:")
    show(rec.popular(3))  # proactive suggestion so the user doesn't face a blank prompt
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        low = line.lower()
        if low in ("quit", "exit", "q"):
            break
        elif low in ("help", "?"):
            print(HELP)
        elif low == "list":
            print("\n".join("  " + t for t in rec.movies["title"]))
        elif low == "popular":
            show(rec.popular(5))
        elif low.startswith("more like "):
            res = rec.similar_to(line[10:])
            if res is None:
                print("  I don't know that title. Type 'list' to see what I have.")
            else:
                show(res)
        elif low.startswith("like "):
            title = line[5:]
            if rec._index_of(title) is None:
                print("  I don't know that title. Type 'list' to see what I have.")
            else:
                liked.append(title)
                print("  Noted. Say 'recommend' when you want suggestions.")
        elif low.startswith("dislike "):
            title = line[8:]
            if rec._index_of(title) is None:
                print("  I don't know that title.")
            else:
                disliked.append(title)
                print("  Got it, I'll steer away from things like that.")
        elif low.startswith("mood "):
            exclude = [i for t in liked + disliked if (i := rec._index_of(t)) is not None]
            show(rec.from_mood(line[5:], exclude=exclude))
        elif low == "recommend":
            show(rec.for_profile(liked, disliked))
        else:
            print("  Not sure what you mean. Type 'help'.")
    print("Enjoy the movie!")


if __name__ == "__main__":
    main()
