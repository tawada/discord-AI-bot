import logging

import discord_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    discord_client.run()


if __name__ == "__main__":
    main()
