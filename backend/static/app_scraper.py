import asyncio
import aiohttp
import os
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from urllib.parse import urljoin, quote_plus
import json
from typing import Optional, Dict, List, Any
import random
import re

# Helper function untuk membersihkan dan membandingkan judul
def compare_titles(title_from_db: str, title_from_play_store: str, min_match_words: int = 3) -> bool:
    """
    Compares two app titles with tolerance for minor differences.
    - Ignores case and non-alphanumeric characters.
    - Checks for a minimum number of common words.
    """
    if not title_from_db or not title_from_play_store:
        return False

    # Normalisasi: lowercase, hapus non-alphanumeric, split jadi kata
    clean = lambda t: set(re.sub(r'[^a-z0-9\s]', '', t.lower()).split())
    
    words_db = clean(title_from_db)
    words_ps = clean(title_from_play_store)

    if not words_db or not words_ps:
        return False

    # Hitung kata yang sama
    common_words = words_db.intersection(words_ps)
    
    # Jika judul pendek, periksa apakah semua kata dari DB ada di Play Store
    if len(words_db) <= min_match_words:
        return words_db.issubset(words_ps)
    
    # Jika judul lebih panjang, periksa jumlah kata yang cocok
    return len(common_words) >= min_match_words


class PlayStoreIconScraper:
    BASE_URL = "https://play.google.com"
    
    def __init__(self, max_concurrent_requests: int = 5, cache_dir: str = "icon_cache"):
        self.session = None
        self.ua = UserAgent()
        self.max_concurrent_requests = max_concurrent_requests
        self.cache_dir = cache_dir
        self.cache: Dict[str, Optional[str]] = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.load_cache()

    def load_cache(self):
        cache_file = os.path.join(self.cache_dir, "icon_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Cache file at '{cache_file}' is corrupted. Starting with an empty cache.")
                self.cache = {}

    def save_cache(self):
        cache_file = os.path.join(self.cache_dir, "icon_cache.json")
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    async def init_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def _get_icon_from_page(self, soup: BeautifulSoup) -> Optional[str]:
        """Helper to extract icon URL from a BeautifulSoup object."""
        meta_icon = soup.find('meta', {'property': 'og:image'})
        if meta_icon and meta_icon.get('content'):
            icon_url = meta_icon['content']
            return icon_url.split('=s')[0] + '=s512'
        
        icon_elem = soup.select_one('div[class*="xSyT2c"] img[class*="T75of"]')
        if icon_elem and icon_elem.get('src'):
            icon_url = icon_elem['src']
            return icon_url.split('=s')[0] + '=s512'
            
        return None

    async def get_app_icon_url(self, app_id: str, app_title: str) -> Optional[str]:
        """
        Gets the icon URL. If app_id fails (404), it searches by app_title as a fallback.
        """
        if not app_id or not isinstance(app_id, str):
            return None

        if app_id in self.cache:
            print(f"Cache hit for {app_id}")
            return self.cache[app_id]

        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        try:
            # --- Rencana A: Coba dengan App ID ---
            details_url = f"{self.BASE_URL}/store/apps/details?id={app_id}&hl=en&gl=US"
            print(f"Plan A: Fetching by App ID for '{app_title}' ({app_id})")
            async with self.session.get(details_url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    icon_url = await self._get_icon_from_page(soup)
                    if icon_url:
                        print(f"Success (Plan A) for {app_id}")
                        self.cache[app_id] = icon_url
                        self.save_cache()
                        return icon_url
                
                # Jika status bukan 200 (terutama 404), lanjut ke Rencana B
                print(f"Plan A failed for {app_id} with status {response.status}. Trying Plan B.")

            # --- Rencana B: Cari berdasarkan Judul Aplikasi ---
            search_url = f"{self.BASE_URL}/store/search?q={quote_plus(app_title)}&c=apps&hl=en&gl=US"
            print(f"Plan B: Searching by title for '{app_title}'")
            async with self.session.get(search_url, headers=headers, timeout=20) as response:
                if response.status != 200:
                    print(f"Plan B search failed with status {response.status}")
                    self.cache[app_id] = None
                    self.save_cache()
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Cari hasil pencarian pertama
                first_result = soup.select_one('div[class*="ULeU3b"]') # Container untuk setiap hasil
                if not first_result:
                    print(f"Plan B: No search results found for '{app_title}'")
                    self.cache[app_id] = None
                    self.save_cache()
                    return None

                # Validasi judul
                found_title_elem = first_result.select_one('div[class*="Epkrse"]')
                found_title = found_title_elem.text.strip() if found_title_elem else ""
                
                if compare_titles(app_title, found_title):
                    print(f"Plan B: Title match found! DB: '{app_title}', PS: '{found_title}'")
                    # Ambil ikon dari hasil pencarian
                    icon_elem = first_result.select_one('img[class*="T75of"]')
                    if icon_elem and icon_elem.get('src'):
                        icon_url = icon_elem['src'].split('=s')[0] + '=s512'
                        print(f"Success (Plan B) for {app_id}")
                        self.cache[app_id] = icon_url
                        self.save_cache()
                        return icon_url
                else:
                    print(f"Plan B: Title mismatch. DB: '{app_title}', PS: '{found_title}'")

        except Exception as e:
            print(f"An exception occurred for {app_id} ('{app_title}'): {e}")

        # Jika semua gagal
        print(f"All plans failed for {app_id} ('{app_title}')")
        self.cache[app_id] = None
        self.save_cache()
        return None

    async def get_multiple_app_icons(self, apps_data: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        await self.init_session()
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def get_with_semaphore(app_data: Dict[str, Any]) -> tuple[str, Optional[str]]:
            app_id = app_data.get("app_id")
            app_title = app_data.get("title")
            if not app_id or not app_title:
                return app_id, None

            async with sem:
                await asyncio.sleep(random.uniform(0.5, 1.5))
                icon_url = await self.get_app_icon_url(app_id, app_title)
                return app_id, icon_url
        
        try:
            tasks = [get_with_semaphore(app_data) for app_data in apps_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            icon_urls = {}
            for result in results:
                if isinstance(result, tuple):
                    app_id, icon_url = result
                    if app_id: icon_urls[app_id] = icon_url
                elif isinstance(result, Exception):
                    print(f"A task failed with an exception: {result}")
            
            return icon_urls
        finally:
            await self.close_session()

def get_app_icons(apps_data: List[Dict[str, Any]], max_concurrent_requests: int = 5) -> Dict[str, Optional[str]]:
    """
    High-level function to get icon URLs for a list of apps.
    
    Args:
        apps_data: A list of dictionaries, where each dict must contain 'app_id' and 'title'.
        
    Returns:
        A dictionary mapping each app ID to its icon URL (or None if not found).
    """
    if not apps_data:
        return {}
        
    print(f"Starting smart icon scraping for {len(apps_data)} app(s).")
    scraper = PlayStoreIconScraper(max_concurrent_requests=max_concurrent_requests)
    return asyncio.run(scraper.get_multiple_app_icons(apps_data))

# Example Usage:
if __name__ == '__main__':
    # Contoh data, termasuk kasus yang mungkin gagal
    test_apps_data = [
        {'app_id': 'com.whatsapp', 'title': 'WhatsApp Messenger'},
        {'app_id': 'com.supermegacorp.extremecitygtcarstunts3d', 'title': 'Extreme City GT Car Stunts 3D'}, # App ID fiktif, akan cari by title
        {'app_id': 'com.google.android.gm', 'title': 'Gmail'},
        {'app_id': 'com.nonexistent.app.id12345', 'title': 'Fake App That Does Not Exist'}, # Akan gagal total
    ]

    icons = get_app_icons(test_apps_data, max_concurrent_requests=3)
    print("\n--- Scraping Results ---")
    print(json.dumps(icons, indent=2))
    print("------------------------")