"""CLI launcher for StreamClip Minimalist Web Studio."""

import argparse
import sys
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="StreamClip AI Minimalist Web Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto reload on code change")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" StreamClip Minimalist Studio Running on:")
    print(f" http://{args.host}:{args.port}")
    print(f"=======================================================\n")

    uvicorn.run("app.web.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
