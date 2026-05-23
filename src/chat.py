"""Interactive command-line chatbot.

Simulates a real customer-support conversation in the terminal: type a
customer message, the trained sequential model replies as the support agent.

Commands:
    /greedy   - switch to deterministic greedy decoding (default)
    /sample   - switch to temperature/top-k sampling (more varied replies)
    /reset    - clear the on-screen conversation history
    /help     - show commands
    /quit     - exit

Run:  python -m src.chat
"""
from src import config
from src.inference import ModelNotTrainedError, get_generator

BANNER = r"""
============================================================
  Social Media Customer Support - Reply Assistant (CLI)
  Sequential seq2seq + attention model (no LLM APIs)
============================================================
  Type a customer message and press Enter.
  Commands: /greedy  /sample  /reset  /help  /quit
"""


def main():
    print(BANNER)
    try:
        gen = get_generator()
    except ModelNotTrainedError as exc:
        print(f"[!] {exc}")
        return

    decoding = config.DEFAULT_DECODING
    history = []
    print(f"[model loaded - decoding mode: {decoding}]\n")

    while True:
        try:
            msg = input("Customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not msg:
            continue
        low = msg.lower()
        if low in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        if low == "/help":
            print(BANNER)
            continue
        if low == "/reset":
            history.clear()
            print("[conversation history cleared]\n")
            continue
        if low in ("/greedy", "/sample"):
            decoding = low[1:]
            print(f"[decoding mode -> {decoding}]\n")
            continue

        reply = gen.reply(msg, decoding=decoding)
        history.append(("customer", msg))
        history.append(("support", reply))
        print(f"Support > {reply}\n")


if __name__ == "__main__":
    main()
