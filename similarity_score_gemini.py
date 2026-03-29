"""
Score job similarity between occupations using Google Gemini 3.0 Flash (free).

Features:
- Batch comparison with intelligent caching
- Rate limit handling (15 req/min, 1.5M tokens/day)
- Exponential backoff retry logic
- Similarity matrix generation
- Resume on interruption with incremental cache saves
- Optimized for Gemini 3.0 Flash model

Usage:
    python similarity_score_gemini.py single medical-transcriptionists --top 15
    python similarity_score_gemini.py batch software-developers data-scientists --top 10
"""

import json
import os
import time
import hashlib
import sys
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

load_dotenv()

# Gemini 2.0 Flash - Free, fast, high quality
GEMINI_MODEL = "gemini-3.0-flash"
CACHE_FILE = "similarity_cache.json"
MATRIX_FILE = "similarity_matrix.json"

# Rate limiting settings for Gemini free tier
RATE_LIMIT_DELAY = 0.3  # seconds between API calls
MAX_RETRIES = 5
BASE_BACKOFF = 2  # exponential backoff base

SIMILARITY_PROMPT = """\
You are a job analyst comparing two occupations based on their **Key Duties** and **Education & Skills**.

You will receive descriptions of two occupations. Your task is to:
1. Extract the key duties from each occupation
2. Extract the education and skills requirements from each
3. Compare them for similarity
4. Provide a **Job Similarity Score** from 0-100

Scoring guidelines:
- **90-100: Nearly identical jobs.** Same duties, same education/skills.
  Examples: Software Developer vs Software Engineer, Registered Nurse vs RN
  
- **75-89: Very similar jobs.** 80%+ overlap in duties and requirements.
  Examples: Electrician vs Electrical Technician, Accountant vs Bookkeeper
  
- **60-74: Similar jobs.** Significant overlap in duties (50-75%) and education.
  Examples: Teacher vs Trainer, Graphic Designer vs Web Designer
  
- **40-59: Moderately similar.** Some overlap in skills/education but different core duties.
  Examples: Carpenter vs Electrician, Nurse vs Medical Assistant
  
- **20-39: Somewhat related.** Limited overlap, different fields but adjacent.
  Examples: Data Analyst vs Business Analyst, Marketing Manager vs Sales Manager
  
- **0-19: Unrelated jobs.** Minimal overlap in duties or requirements.
  Examples: Software Developer vs Dentist, Roofer vs Accountant

Respond with ONLY a JSON object in this exact format, no other text:
{
  "similarity_score": <0-100>,
  "duty_overlap_percent": <0-100>,
  "skill_overlap_percent": <0-100>,
  "shared_duties": ["<duty1>", "<duty2>", ...],
  "shared_skills": ["<skill1>", "<skill2>", ...],
  "key_differences": ["<difference1>", "<difference2>", ...],
  "rationale": "<2-3 sentences explaining the score>"
}
"""


class SimilarityCache:
    """Manages caching with incremental saving and resume capability."""
    
    def __init__(self, cache_file=SIMILARITY_CACHE):
        self.cache_file = cache_file
        self.cache = {}
        self.comparisons_made = 0
        self.comparisons_cached = 0
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
                print(f"✅ Cache loaded: {len(self.cache)} entries\n")
            except json.JSONDecodeError:
                print(f"⚠️  Cache file corrupted, starting fresh\n")
                self.cache = {}
        else:
            print(f"📝 New cache file will be created: {self.cache_file}\n")
    
    def save(self):
        """Save cache to disk immediately."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_key(self, slug1, slug2):
        """Generate canonical cache key (order-independent)."""
        pair = tuple(sorted([slug1, slug2]))
        return hashlib.md5("|".join(pair).encode()).hexdigest()
    
    def get(self, slug1, slug2):
        """Retrieve cached comparison."""
        key = self.get_key(slug1, slug2)
        if key in self.cache:
            self.comparisons_cached += 1
            return self.cache[key]
        return None
    
    def set(self, slug1, slug2, result):
        """Store comparison in cache and save immediately."""
        key = self.get_key(slug1, slug2)
        self.cache[key] = {
            "slug1": slug1,
            "slug2": slug2,
            "timestamp": time.time(),
            **result
        }
        self.comparisons_made += 1
        self.save()  # ✅ Save after each comparison
    
    def get_stats(self):
        """Return cache statistics."""
        return {
            "total_cached": len(self.cache),
            "comparisons_made": self.comparisons_made,
            "comparisons_cached": self.comparisons_cached,
        }


def load_csv_data():
    """Load job data from CSV."""
    
    jobs = {}
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get('slug') or row.get('Slug')
            if not slug:
                continue
            
            jobs[slug] = {
                'title': row.get('title') or row.get('Title') or 'Unknown',
                'category': row.get('category') or row.get('Category') or 'General',
                'description': row.get('description') or row.get('Description') or '',
                'url': row.get('url') or row.get('URL') or '',
            }
    
    print(f"📚 Loaded {len(jobs)} jobs from CSV\n")
    return jobs


def compare_jobs_with_gemini(model, job1_title, job1_desc, job2_title, job2_desc):
    """
    Compare two jobs using Gemini with intelligent rate limiting and retries.
    
    Rate Limiting Strategy:
    - Base delay: 0.3s between calls
    - Quota errors: exponential backoff (1s, 2s, 4s, 8s, 16s)
    - Server errors: 5s, 10s, 15s delays
    """
    
    comparison_text = f"""
Job 1: {job1_title}
Description: {job1_desc[:400]}

Job 2: {job2_title}
Description: {job2_desc[:400]}

Compare these jobs.
"""
    
    full_prompt = f"{SIMILARITY_PROMPT}\n\n{comparison_text}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(full_prompt)
            content = response.text.strip()
            
            # Strip markdown fences
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            result = json.loads(content)
            
            # ✅ Rate limiting after successful call
            if attempt < MAX_RETRIES - 1:
                time.sleep(RATE_LIMIT_DELAY)
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"\n  ❌ JSON parse error: {e}")
            print(f"  Response was: {content[:100]}...")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BASE_BACKOFF ** attempt)
            continue
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # ✅ QUOTA/RATE LIMIT - Exponential backoff
            if any(x in error_msg for x in ["quota", "rate", "429", "too_many_requests"]):
                wait_time = BASE_BACKOFF ** attempt
                print(f"\n  ⏳ RATE LIMIT HIT!")
                print(f"  Gemini free tier quota exceeded")
                print(f"  Waiting {wait_time}s before retry (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                continue
            
            # SERVER ERROR - Longer delays
            if any(x in error_msg for x in ["500", "502", "503", "service"]):
                wait_time = 5 * (attempt + 1)
                print(f"\n  ⚠️  SERVER ERROR ({error_msg[:30]})")
                print(f"  Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # AUTHENTICATION ERROR
            if "api key" in error_msg or "auth" in error_msg:
                print(f"\n  ❌ Authentication failed: {e}")
                raise
            
            # OTHER ERRORS
            if attempt == MAX_RETRIES - 1:
                raise
            
            print(f"\n  ⚠️  Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(BASE_BACKOFF ** attempt)
    
    raise Exception(f"Failed after {MAX_RETRIES} retries")


def find_similar_jobs(target_slug, top_n=10):
    """Find similar jobs for a target job."""
    
    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Load data
    jobs = load_csv_data()
    cache = SimilarityCache()
    
    if target_slug not in jobs:
        print(f"❌ Job '{target_slug}' not found in CSV")
        print(f"Available jobs: {', '.join(list(jobs.keys())[:5])}...")
        return
    
    target_job = jobs[target_slug]
    target_title = target_job['title']
    target_desc = target_job['description']
    
    print(f"🔍 Finding similar jobs to: {target_title}")
    print(f"📊 Model: {GEMINI_MODEL}")
    print(f"⏱️  Rate limit: 0.3s between calls")
    print(f"💾 Cache file: {SIMILARITY_CACHE}")
    print(f"Comparing against {len(jobs)} occupations...\n")
    
    similarities = []
    start_time = time.time()
    
    for i, (slug, job_data) in enumerate(jobs.items()):
        if slug == target_slug:
            continue
        
        print(f"  [{i+1:3d}/{len(jobs)}] {job_data['title']:<40}", end=" ", flush=True)
        
        try:
            # ✅ Check cache first
            cached = cache.get(target_slug, slug)
            if cached:
                result = {k: v for k, v in cached.items() if k not in ["slug1", "slug2", "timestamp"]}
                print("(cached)")
            else:
                # Call Gemini API
                result = compare_jobs_with_gemini(
                    model,
                    target_title,
                    target_desc,
                    job_data['title'],
                    job_data['description']
                )
                # ✅ Save to cache immediately
                cache.set(target_slug, slug, result)
                print(f"Score: {result['similarity_score']:3d}/100")
            
            similarities.append({
                "slug": slug,
                "title": job_data['title'],
                "category": job_data['category'],
                **result
            })
        
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
    
    elapsed_time = time.time() - start_time
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # ✅ Display results
    print(f"\n{'='*100}")
    print(f"Top {top_n} Most Similar Jobs to: {target_title}")
    print(f"{'='*100}\n")
    
    for idx, job in enumerate(similarities[:top_n], 1):
        print(f"{idx:2d}. {job['title']:<50} (Score: {job['similarity_score']:3d}/100)")
        print(f"    Category: {job['category']:<20} Duty: {job['duty_overlap_percent']:3d}% | Skill: {job['skill_overlap_percent']:3d}%")
        print(f"    Skills: {', '.join(job['shared_skills'][:2])}")
        print(f"    Rationale: {job['rationale'][:70]}...\n")
    
    # ✅ Save results to file
    output_file = f"similarity_results_{target_slug}.json"
    with open(output_file, 'w') as f:
        json.dump(similarities, f, indent=2)
    print(f"✅ Results saved to: {output_file}")
    
    # ✅ Print statistics
    stats = cache.get_stats()
    print(f"\n{'='*100}")
    print(f"📊 STATISTICS:")
    print(f"{'='*100}")
    print(f"⏱️  Time elapsed: {elapsed_time:.1f}s")
    print(f"📝 Total cached: {stats['total_cached']}")
    print(f"🆕 New API calls: {stats['comparisons_made']}")
    print(f"♻️  From cache: {stats['comparisons_cached']}")
    print(f"💾 Cache file: {SIMILARITY_CACHE}")
    print(f"📄 Results file: {output_file}")
    print(f"⏱️  Daily limit: 1.5M tokens (track at console.cloud.google.com)")


def batch_compare_jobs(target_slugs, top_n=10):
    """Compare multiple jobs in batch."""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    jobs = load_csv_data()
    cache = SimilarityCache()
    
    batch_start = time.time()
    
    for target_slug in target_slugs:
        if target_slug not in jobs:
            print(f"⚠️  Job '{target_slug}' not found, skipping...")
            continue
        
        target_job = jobs[target_slug]
        print(f"\n🔍 Analyzing: {target_job['title']}\n")
        
        similarities = []
        
        for i, (slug, job_data) in enumerate(jobs.items()):
            if slug == target_slug:
                continue
            
            print(f"  [{i+1:3d}] {job_data['title']:<40}", end=" ", flush=True)
            
            try:
                cached = cache.get(target_slug, slug)
                if cached:
                    result = {k: v for k, v in cached.items() if k not in ["slug1", "slug2", "timestamp"]}
                    print("✓")
                else:
                    result = compare_jobs_with_gemini(
                        model,
                        target_job['title'],
                        target_job['description'],
                        job_data['title'],
                        job_data['description']
                    )
                    cache.set(target_slug, slug, result)
                    print("✓")
                
                similarities.append({
                    "slug": slug,
                    "title": job_data['title'],
                    "category": job_data['category'],
                    **result
                })
            
            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"Top {top_n} Similar Jobs:")
        print(f"{'='*80}\n")
        
        for idx, job in enumerate(similarities[:top_n], 1):
            print(f"{idx:2d}. {job['title']:<50} ({job['similarity_score']:3d}/100)")
    
    batch_elapsed = time.time() - batch_start
    print(f"\n✅ Batch complete in {batch_elapsed:.1f}s")
    
    # ✅ Final statistics
    stats = cache.get_stats()
    print(f"\n{'='*80}")
    print(f"📊 FINAL STATISTICS:")
    print(f"{'='*80}")
    print(f"Total comparisons: {stats['comparisons_made'] + stats['comparisons_cached']}")
    print(f"New API calls: {stats['comparisons_made']}")
    print(f"From cache: {stats['comparisons_cached']}")
    print(f"Cache efficiency: {(stats['comparisons_cached']/(stats['comparisons_made']+stats['comparisons_cached'])*100):.1f}%")
    print(f"Cache file: {SIMILARITY_CACHE} ({len(cache.cache)} entries)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find similar jobs using CSV and Gemini")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    single = subparsers.add_parser("single", help="Find similar jobs to one occupation")
    single.add_argument("job_slug", help="Job slug from CSV")
    single.add_argument("--top", type=int, default=10, help="Top N results")
    
    batch = subparsers.add_parser("batch", help="Compare multiple jobs")
    batch.add_argument("job_slugs", nargs="+", help="Job slugs")
    batch.add_argument("--top", type=int, default=10, help="Top N results")
    
    cache_cmd = subparsers.add_parser("cache", help="Manage cache")
    cache_cmd.add_argument("--stats", action="store_true", help="Show cache stats")
    cache_cmd.add_argument("--clear", action="store_true", help="Clear cache")
    
    args = parser.parse_args()
    
    if args.command == "single":
        find_similar_jobs(args.job_slug, top_n=args.top)
    elif args.command == "batch":
        batch_compare_jobs(args.job_slugs, top_n=args.top)
    elif args.command == "cache":
        cache = SimilarityCache()
        stats = cache.get_stats()
        if args.stats:
            print(f"\n📊 Cache Statistics:")
            print(f"   Total entries: {len(cache.cache)}")
            print(f"   File: {SIMILARITY_CACHE}")
            print(f"   File size: {os.path.getsize(SIMILARITY_CACHE) / 1024:.1f} KB")
        elif args.clear:
            cache.cache = {}
            cache.save()
            print("✅ Cache cleared")
    else:
        parser.print_help()
