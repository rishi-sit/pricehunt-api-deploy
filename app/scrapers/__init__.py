from .base import BaseScraper
from .amazon import AmazonScraper
from .amazon_fresh import AmazonFreshScraper
from .flipkart import FlipkartScraper
from .flipkart_minutes import FlipkartMinutesScraper
from .zepto import ZeptoScraper
from .instamart import InstamartScraper
from .blinkit import BlinkitScraper
from .bigbasket import BigBasketScraper
from .jiomart_quick import JioMartQuickScraper
from .jiomart import JioMartScraper

__all__ = [
    "BaseScraper",
    "AmazonScraper",
    "AmazonFreshScraper",
    "FlipkartScraper",
    "FlipkartMinutesScraper",
    "ZeptoScraper",
    "InstamartScraper",
    "BlinkitScraper",
    "BigBasketScraper",
    "JioMartQuickScraper",
    "JioMartScraper",
]

