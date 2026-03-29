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
RATE_LIMIT_RPM = 15  # Requests per minute
RATE_LIMIT_TPD = 1_500_000  # Tokens per day
DELAY_BETWEEN_REQUESTS = 4  # 4 second delay = ~15 req/min

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


class RateLimiter:
    """Handles rate limiting for Gemini API."""
    
    def __init__(self, rpm=RATE_LIMIT_RPM):
        self.rpm = rpm
        self.delay = 60 / rpm  # Calculate delay between requests
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()


class SimilarityCache:
    """Manages caching of job similarity comparisons."""
    
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = {}
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
                print(f"✅ Loaded {len(self.cache)} cached comparisons")
            except json.JSONDecodeError:
                print("⚠️  Cache file corrupted, starting fresh")
                self.cache = {}
        else:
            self.cache = {}
    
    def save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def get_key(self, slug1, slug2):
        """Generate a canonical cache key (order-independent)."""
        pair = tuple(sorted([slug1, slug2]))
        return hashlib.md5("|".join(pair).encode()).hexdigest()
    
    def get(self, slug1, slug2):
        """Retrieve a cached comparison."""
        key = self.get_key(slug1, slug2)
        return self.cache.get(key)
    
    def set(self, slug1, slug2, result):
        """Store a comparison in cache."""
        key = self.get_key(slug1, slug2)
        self.cache[key] = {
            "slug1": slug1,
            "slug2": slug2,
            "timestamp": time.time(),
            **result
        }
        self.save()  # Save after each entry
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.save()
        print("✅ Cache cleared")


def get_job_descriptions_from_markdown(slug):
    """Extract Key Duties and Education & Skills from markdown file."""
    md_path = f"pages/{slug}.md"
    
    if not os.path.exists(md_path):
        return None
    
    try:
        with open(md_path) as f:
            content = f.read()
        
        # Extract Key Duties section
        duties_start = content.find("**Key Duties**")
        duties_end = content.find("\n**", duties_start + 1) if duties_start != -1 else -1
        duties = content[duties_start:duties_end] if duties_start != -1 else "Key Duties not found"
        
        # Extract Education & Skills section
        skills_start = content.find("**Education & Skills**")
        skills_end = len(content)
        skills = content[skills_start:skills_end] if skills_start != -1 else "Education & Skills not found"
        
        return {
            "slug": slug,
            "duties": duties,
            "skills": skills
        }
    except Exception as e:
        print(f"❌ Error reading {md_path}: {e}")
        return None


def compare_jobs_gemini(model, job1_data, job2_data, rate_limiter, max_retries=5):
    """
    Compare two jobs using Gemini 2.0 Flash with exponential backoff.
    
    Args:
        model: Gemini model instance
        job1_data: First job data dict
        job2_data: Second job data dict
        rate_limiter: RateLimiter instance
        max_retries: Max retry attempts on rate limit
    
    Returns:
        dict: Similarity score result
    """
    
    comparison_text = f"""
Job 1: {job1_data['slug']}
{job1_data['duties']}
{job1_data['skills']}

Job 2: {job2_data['slug']}
{job2_data['duties']}
{job2_data['skills']}

Compare these two jobs based on Key Duties and Education & Skills.
"""
    
    full_prompt = f"{SIMILARITY_PROMPT}\n\n{comparison_text}"
    
    for attempt in range(max_retries):
        try:
            # Respect rate limits
            rate_limiter.wait_if_needed()
            
            # Call Gemini API
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                )
            )
            
            content = response.text.strip()
            
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            
            result = json.loads(content)
            
            # Validate result structure
            required_keys = ["similarity_score", "duty_overlap_percent", "skill_overlap_percent", 
                           "shared_duties", "shared_skills", "key_differences", "rationale"]
            if not all(key in result for key in required_keys):
                raise ValueError("Missing required keys in response")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f" [JSON Error: {e}]", end="", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit error
            if "quota" in error_msg or "rate" in error_msg or "429" in error_msg or "resource_exhausted" in error_msg:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"\n⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...", flush=True)
                time.sleep(wait_time)
                continue
            
            # Check for server errors
            if "500" in error_msg or "service" in error_msg or "unavailable" in error_msg:
                wait_time = 5 * (attempt + 1)
                print(f"\n⚠️  Server error. Waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            
            if attempt == max_retries - 1:
                raise
            
            print(f"\n❌ Error (attempt {attempt + 1}): {e}. Retrying...", flush=True)
            time.sleep(2 ** attempt)
    
    raise Exception(f"Failed after {max_retries} attempts")


def compare_single_job(target_slug, top_n=10):
    """Compare a single job against all occupations."""
    
    # Configure Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        print("   Set it via: export GEMINI_API_KEY='your_key'")
        print("   Or add to .env: GEMINI_API_KEY=your_key")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    with open("occupations.json") as f:
        all_occupations = json.load(f)
    
    target_data = get_job_descriptions_from_markdown(target_slug)
    if not target_data:
        print(f"❌ Could not find markdown for {target_slug}")
        return
    
    target_title = next((occ["title"] for occ in all_occupations if occ["slug"] == target_slug), target_slug)
    print(f"\n🔍 Analyzing: {target_title} ({target_slug})")
    print(f"Comparing against {len(all_occupations)} occupations...")
    print(f"⏱️  Rate limit: 15 requests/min, 1.5M tokens/day\n")
    
    cache = SimilarityCache()
    rate_limiter = RateLimiter(rpm=RATE_LIMIT_RPM)
    
    similarities = []
    cached_count = 0
    api_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for i, occ in enumerate(all_occupations):
        slug = occ["slug"]
        
        # Skip the target job itself
        if slug == target_slug:
            continue
        
        comp_data = get_job_descriptions_from_markdown(slug)
        if not comp_data:
            continue
        
        print(f"  [{i+1}/{len(all_occupations)}] {occ['title']}...", end=" ", flush=True)
        
        try:
            # Check cache first
            cached = cache.get(target_slug, slug)
            if cached:
                result = {k: v for k, v in cached.items() if k not in ["slug1", "slug2", "timestamp"]}
                print("(cached)")
                cached_count += 1
            else:
                result = compare_jobs_gemini(model, target_data, comp_data, rate_limiter)
                cache.set(target_slug, slug, result)
                print(f"Score: {result['similarity_score']}/100")
                api_count += 1
            
            similarities.append({
                "slug": slug,
                "title": occ["title"],
                "category": occ["category"],
                **result
            })
        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1
    
    elapsed = time.time() - start_time
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top {top_n} Most Similar Jobs to: {target_title}")
    print(f"{'='*80}\n")
    
    for idx, job in enumerate(similarities[:top_n], 1):
        print(f"{idx}. {job['title']} (Similarity: {job['similarity_score']}/100)")
        print(f"   Slug: {job['slug']}")
        print(f"   Category: {job['category']}")
        print(f"   Duty Overlap: {job['duty_overlap_percent']}% | Skill Overlap: {job['skill_overlap_percent']}%")
        print(f"   Shared Skills: {', '.join(job['shared_skills'][:3])}...")
        print(f"   Key Differences: {', '.join(job['key_differences'][:2])}...")
        print(f"   Rationale: {job['rationale'][:100]}...")
        print()
    
    # Save results
    output_file = f"similarity_results_{target_slug}.json"
    with open(output_file, "w") as f:
        json.dump(similarities, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Full results saved to: {output_file}")
    print(f"📊 Statistics:")
    print(f"   - API calls: {api_count}")
    print(f"   - From cache: {cached_count}")
    print(f"   - Errors: {error_count}")
    print(f"   - Total time: {elapsed:.1f}s")
    print(f"   - Cache file: {CACHE_FILE} ({len(cache.cache)} total entries)")
    print(f"{'='*80}\n")


def batch_compare_jobs(target_slugs, top_n=10):
    """Compare multiple jobs."""
    
    # Configure Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    with open("occupations.json") as f:
        all_occupations = json.load(f)
    
    cache = SimilarityCache()
    rate_limiter = RateLimiter(rpm=RATE_LIMIT_RPM)
    
    all_results = {}
    
    for target_slug in target_slugs:
        target_data = get_job_descriptions_from_markdown(target_slug)
        if not target_data:
            print(f"❌ Could not find markdown for {target_slug}")
            continue
        
        target_title = next((occ["title"] for occ in all_occupations if occ["slug"] == target_slug), target_slug)
        print(f"\n🔍 Analyzing: {target_title}")
        
        similarities = []
        
        for i, occ in enumerate(all_occupations):
            slug = occ["slug"]
            
            if slug == target_slug:
                continue
            
            comp_data = get_job_descriptions_from_markdown(slug)
            if not comp_data:
                continue
            
            print(f"  [{i+1}] {occ['title']}...", end=" ", flush=True)
            
            try:
                cached = cache.get(target_slug, slug)
                if cached:
                    result = {k: v for k, v in cached.items() if k not in ["slug1", "slug2", "timestamp"]}
                    print("(cached)")
                else:
                    result = compare_jobs_gemini(model, target_data, comp_data, rate_limiter)
                    cache.set(target_slug, slug, result)
                    print(f"✓")
                
                similarities.append({
                    "slug": slug,
                    "title": occ["title"],
                    "category": occ["category"],
                    **result
                })
            except Exception as e:
                print(f"ERROR: {e}")
        
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        all_results[target_slug] = similarities
        
        # Display top N
        print(f"\n{'='*80}")
        print(f"Top {top_n} Most Similar to: {target_title}")
        print(f"{'='*80}\n")
        
        for idx, job in enumerate(similarities[:top_n], 1):
            print(f"{idx}. {job['title']} ({job['similarity_score']}/100)")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find similar jobs using Google Gemini 2.0 Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single job
  python similarity_score_gemini.py single medical-transcriptionists --top 15
  
  # Batch compare multiple jobs
  python similarity_score_gemini.py batch software-developers data-scientists --top 10
  
  # Cache management
  python similarity_score_gemini.py cache --stats
  python similarity_score_gemini.py cache --clear
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    single = subparsers.add_parser("single", help="Compare single job")
    single.add_argument("job_slug", help="Job slug to analyze")
    single.add_argument("--top", type=int, default=10, help="Number of similar jobs to show")
    
    batch = subparsers.add_parser("batch", help="Compare multiple jobs")
    batch.add_argument("job_slugs", nargs="+", help="Job slugs to compare")
    batch.add_argument("--top", type=int, default=10, help="Number of similar jobs to show")
    
    cache_cmd = subparsers.add_parser("cache", help="Manage cache")
    cache_cmd.add_argument("--stats", action="store_true", help="Show cache statistics")
    cache_cmd.add_argument("--clear", action="store_true", help="Clear all cached comparisons")
    
    args = parser.parse_args()
    
    if args.command == "single":
        compare_single_job(args.job_slug, top_n=args.top)
    elif args.command == "batch":
        batch_compare_jobs(args.job_slugs, top_n=args.top)
    elif args.command == "cache":
        cache = SimilarityCache()
        if args.clear:
            cache.clear()
        elif args.stats:
            print(f"\n📊 Cache Statistics:")
            print(f"   Total cached comparisons: {len(cache.cache)}")
            print(f"   Cache file: {CACHE_FILE}")
            if os.path.exists(CACHE_FILE):
                size_kb = os.path.getsize(CACHE_FILE) / 1024
                print(f"   Cache size: {size_kb:.1f} KB")
            print()
        else:
            print(f"✅ Cache loaded: {len(cache.cache)} comparisons")
    else:
        parser.print_help()
