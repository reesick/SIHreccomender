import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Career Advisor Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Global data storage
careers_df = None
colleges_df = None

# Simple feedback storage
feedback_data = []

# Pydantic models
class StudentAnswers(BaseModel):
    student_id: Optional[str] = None
    education_level: Optional[str] = None  # "10th" or "12th"
    answers: Dict[str, Any]

class CareerRecommendation(BaseModel):
    stream: str
    degree: str
    career: str
    score: float
    reason: str

class CollegeRecommendation(BaseModel):
    college_name: str
    district: str
    course: str
    cutoff: Optional[str]
    reason: str

class RecommendationResponse(BaseModel):
    student_id: Optional[str]
    timestamp: str
    careers: List[CareerRecommendation]
    colleges: List[CollegeRecommendation]

class FeedbackRequest(BaseModel):
    student_id: Optional[str] = None
    recommended_careers: List[str] = []
    selected_career: Optional[str] = None
    recommended_colleges: List[str] = []
    selected_college: Optional[str] = None

# Helper functions
def load_data():
    """Load CSV data at startup"""
    global careers_df, colleges_df
    
    try:
        careers_df = pd.read_csv("data/careers.csv")
        colleges_df = pd.read_csv("data/jk_colleges_rows.csv")
        logger.info(f"Loaded {len(careers_df)} careers and {len(colleges_df)} colleges")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def process_student_answers(answers: Dict[str, Any], education_level: Optional[str] = None) -> Dict[str, Any]:
    """Convert raw answers to structured profile"""
    
    # Extract interest streams from q1
    interest_streams = [answers.get("q1", "")]
    
    # Extract job preference from q2
    job_pref = answers.get("q2", "").replace(" job", "").replace("Government job", "Government")
    
    # Extract math level from q4
    math_mapping = {
        "MathsComfortable_High": 2,
        "MathsComfortable_Medium": 1,
        "MathsComfortable_Low": 0
    }
    math_level = math_mapping.get(answers.get("q4", ""), 1)
    
    # Extract work type from q5
    work_type = answers.get("q5", "").replace("PreferredWorkType_", "")
    
    # Extract languages from q6
    languages = answers.get("q6", [])
    if isinstance(languages, str):
        languages = [languages]
    
    # Extract boolean preferences
    budget_constraint = answers.get("q7", "") == "BudgetConstraint_Yes"
    relocate = answers.get("q8", "") == "WillingToRelocate_Yes"
    
    # Extract skills from q9
    skills = answers.get("q9", [])
    if isinstance(skills, str):
        skills = [skills]
    
    # Extract goal from q10
    goal = answers.get("q10", "").replace("LongTermGoal_", "")
    
    return {
        "education_level": education_level,
        "interest_streams": interest_streams,
        "job_preference": job_pref,
        "location_preference": answers.get("q3", ""),
        "math_level": math_level,
        "work_type": work_type,
        "languages": languages,
        "budget_constraint": budget_constraint,
        "relocate_willing": relocate,
        "skills": skills,
        "goal": goal
    }

def score_careers_for_student(student_profile: Dict[str, Any]) -> pd.DataFrame:
    """Score careers based on student profile with weighted features"""
    
    filtered_df = careers_df.copy()
    filtered_df['score'] = 0.0
    
    # Education level filtering (10th vs 12th)
    education_level = student_profile.get("education_level")
    if education_level == "10th":
        # Include vocational and degree programs for 10th students
        pass  # No filtering needed
    elif education_level == "12th":
        # Only degree programs for 12th students (exclude basic vocational certificates)
        vocational_certs = ['Certificate Auto', 'Certificate Electrical', 'Certificate Plumbing', 
                           'Certificate Welding', 'Certificate Handicraft', 'Certificate Cooking']
        filtered_df = filtered_df[~filtered_df['degree'].isin(vocational_certs)]
    
    # Feature Weighting System
    
    # 1. Stream Interest (30% weight)
    interest_streams = student_profile.get("interest_streams", [])
    if interest_streams and interest_streams[0]:
        stream_match = filtered_df['stream'].isin(interest_streams)
        filtered_df.loc[stream_match, 'score'] += 30
    
    # 2. Skills Match (25% weight)
    student_skills = student_profile.get("skills", [])
    for skill in student_skills:
        skill_variations = {
            'Computer': ['Computer', 'Programming', 'Technical Skills'],
            'BasicMaths': ['Mathematics', 'Analysis'],
            'Laboratory': ['Laboratory Skills', 'Lab'],
            'Communication': ['Communication'],
            'Handicraft': ['Creativity', 'Manual Dexterity']
        }
        
        search_terms = skill_variations.get(skill, [skill])
        for term in search_terms:
            skill_match = filtered_df['skills_needed'].str.contains(term, na=False, case=False)
            filtered_df.loc[skill_match, 'score'] += 25 / len(search_terms)
    
    # 3. Job Preference (20% weight)
    job_pref = student_profile.get("job_preference", "")
    if job_pref == "Government":
        job_match = filtered_df['job_type'].isin(['Government', 'Both'])
        filtered_df.loc[job_match, 'score'] += 20
    elif job_pref == "Private":
        job_match = filtered_df['job_type'].isin(['Private', 'Both'])
        filtered_df.loc[job_match, 'score'] += 20
    elif job_pref == "Entrepreneur":
        # Favor careers with high growth prospects for entrepreneurial mindset
        entrepreneur_match = filtered_df['growth_prospects'] == 'High'
        filtered_df.loc[entrepreneur_match, 'score'] += 15
    
    # 4. Math/Technical Level (15% weight)
    math_level = student_profile.get("math_level", 1)
    if math_level >= 2:  # High math comfort
        math_careers = filtered_df['skills_needed'].str.contains('Mathematics|Technical|Engineering', na=False, case=False)
        filtered_df.loc[math_careers, 'score'] += 15
    elif math_level <= 0:  # Low math comfort
        non_math_careers = ~filtered_df['skills_needed'].str.contains('Mathematics|Technical', na=False, case=False)
        filtered_df.loc[non_math_careers, 'score'] += 10
    
    # 5. Work Type Preference (10% weight)
    work_type = student_profile.get("work_type", "")
    work_type_mapping = {
        'Office': ['Management', 'Administrative', 'Analysis'],
        'Lab': ['Laboratory', 'Research', 'Technical'],
        'Field': ['Field', 'Project', 'Survey'],
        'Creative': ['Creative', 'Design', 'Art']
    }
    
    if work_type in work_type_mapping:
        for work_term in work_type_mapping[work_type]:
            work_match = filtered_df['skills_needed'].str.contains(work_term, na=False, case=False)
            filtered_df.loc[work_match, 'score'] += 10 / len(work_type_mapping[work_type])
    
    # Normalize scores to 0-1 range
    if filtered_df['score'].max() > 0:
        filtered_df['score'] = filtered_df['score'] / 100.0
    
    # Return top 30 careers sorted by score
    return filtered_df.sort_values('score', ascending=False).head(30)

def filter_colleges_for_careers(career_suggestions: List[Dict], student_profile: Dict[str, Any]) -> pd.DataFrame:
    """Filter colleges based on career recommendations and student profile"""
    
    filtered_df = colleges_df.copy()
    
    # Extract degree requirements from career suggestions
    degrees_needed = []
    streams_needed = []
    for career in career_suggestions:
        degree = career.get('degree', '')
        stream = career.get('stream', '')
        if degree:
            degrees_needed.append(degree)
        if stream:
            streams_needed.append(stream)
    
    # Filter colleges that offer these degrees/streams
    if degrees_needed or streams_needed:
        # Use Programs_and_Courses_Offered_Combined column for course matching
        course_mask = pd.Series([False] * len(filtered_df))
        
        for degree in degrees_needed:
            # Match degree names (B.Tech, B.Sc, etc.)
            degree_match = filtered_df['Programs_and_Courses_Offered_Combined'].str.contains(degree, na=False, case=False)
            course_mask = course_mask | degree_match
        
        for stream in streams_needed:
            # Match stream names in Streams column
            stream_match = filtered_df['Streams'].str.contains(stream, na=False, case=False)
            course_mask = course_mask | stream_match
        
        if course_mask.any():
            filtered_df = filtered_df[course_mask]
    
    # Filter by government preference using Affiliation column
    job_pref = student_profile.get("job_preference", "")
    if job_pref == "Government":
        # Look for government-affiliated colleges
        govt_mask = (filtered_df['Affiliation'].str.contains('Government|University of Kashmir|University of Jammu|Central University', na=False, case=False))
        filtered_df = filtered_df[govt_mask]
    
    # Filter by location preference if willing to relocate
    location_pref = student_profile.get("location_preference", "")
    if location_pref and not student_profile.get("relocate_willing", True):
        # Filter by location (district)
        location_match = filtered_df['Location'].str.contains(location_pref, na=False, case=False)
        if location_match.any():
            filtered_df = filtered_df[location_match]
    
    # Filter by budget constraint
    if student_profile.get("budget_constraint", False):
        # Check for colleges with no fees (government colleges) or reasonable fees
        # Many government colleges have '-' or 'Not NIRF ranked' in fees columns, so we'll prioritize government colleges
        govt_mask = (filtered_df['Affiliation'].str.contains('Government|University of Kashmir|University of Jammu', na=False, case=False))
        if govt_mask.any():
            filtered_df = filtered_df[govt_mask]
    
    return filtered_df.head(40)

def call_gemini_for_careers(student_profile: Dict, careers_data: pd.DataFrame) -> List[Dict]:
    """Call Gemini to get career recommendations"""
    
    # Prepare careers data for prompt (include scores if available)
    careers_list = careers_data.to_dict('records')
    
    system_prompt = "You are a career advisor. Output ONLY valid JSON, no other text or markdown."
    
    user_prompt = f"""
Student Profile: {json.dumps(student_profile)}

Available Careers (with relevance scores): {json.dumps(careers_list[:20])}

Task: Recommend top 5 careers matching this student. Consider their education level, interests, skills, job preferences, math level, and work type preference. Use the relevance scores as guidance.

Output exactly this format:
{{
  "careers": [
    {{"stream": "Science", "degree": "B.Tech CSE", "career": "Software Developer", "score": 0.85, "reason": "explanation in max 50 words"}}
  ]
}}
"""
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=2048,
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        logger.info(f"Gemini careers response: {response_text}")
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        logger.info(f"Parsed Gemini result: {result}")
        return result.get('careers', [])
        
    except Exception as e:
        logger.error(f"Error calling Gemini for careers: {e}")
        # Fallback: return sample careers based on student profile
        stream = student_profile.get("interest_streams", ["Science"])[0] if student_profile.get("interest_streams") else "Science"
        return [{
            "stream": stream,
            "degree": f"B.{stream[:3]}",
            "career": f"{stream} Professional",
            "score": 0.7,
            "reason": "Basic recommendation due to API error - please try again"
        }]

def call_gemini_for_colleges(career_suggestions: List[Dict], colleges_data: pd.DataFrame, student_profile: Dict) -> List[Dict]:
    """Call Gemini to get college recommendations"""
    
    # Prepare colleges data for prompt - use relevant columns from new schema
    colleges_list = []
    for _, college in colleges_data.iterrows():
        college_info = {
            "college_name": college.get('College_Name', ''),
            "location": college.get('Location', ''),
            "affiliation": college.get('Affiliation', ''),
            "streams": college.get('Streams', ''),
            "programs": college.get('Programs_and_Courses_Offered_Combined', ''),
            "median_package": college.get('Median_Package_(Latest,_INR_LPA)', ''),
            "contact": college.get('Contact Number', ''),
            "id": college.get('id', '')
        }
        colleges_list.append(college_info)
    
    system_prompt = "You are a college advisor for J&K students. Output ONLY valid JSON, no other text or markdown."
    
    user_prompt = f"""
Career Recommendations: {json.dumps(career_suggestions)}

Available Colleges: {json.dumps(colleges_list[:25])}

Student Preferences: {json.dumps({
    "education_level": student_profile.get("education_level"),
    "budget_constraint": student_profile.get("budget_constraint"),
    "relocate_willing": student_profile.get("relocate_willing"),
    "job_preference": student_profile.get("job_preference"),
    "location_preference": student_profile.get("location_preference")
})}

Task: Recommend top 5 colleges that offer courses matching these careers. Consider program availability, college affiliation, location, and student preferences.

Output exactly this format:
{{
  "colleges": [
    {{"college_name": "SP College", "district": "Srinagar", "course": "B.Tech CSE", "cutoff": "75%", "reason": "explanation in max 30 words"}}
  ]
}}
"""
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\n{user_prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=2048,
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        logger.info(f"Gemini colleges response: {response_text}")
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        return result.get('colleges', [])
        
    except Exception as e:
        logger.error(f"Error calling Gemini for colleges: {e}")
        # Fallback: return sample colleges
        available_colleges = colleges_data.head(3).to_dict('records')
        fallback_colleges = []
        for college in available_colleges:
            programs = college.get('Programs_and_Courses_Offered_Combined', '').split(',')
            fallback_colleges.append({
                "college_name": college.get('College_Name', 'Unknown College'),
                "district": college.get('Location', 'Unknown'),
                "course": programs[0].strip() if programs else 'General',
                "cutoff": 'NA',
                "reason": "Fallback recommendation due to API error"
            })
        return fallback_colleges

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()

@app.get("/")
async def root():
    return {"message": "Career Advisor Backend is running", "status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(payload: StudentAnswers):
    """Main endpoint for career and college recommendations"""
    
    try:
        logger.info(f"Processing request for student: {payload.student_id}")
        
        # Step 1: Process student answers
        student_profile = process_student_answers(payload.answers, payload.education_level)
        logger.info(f"Student profile: {student_profile}")
        
        # Step 2: Get career recommendations using scoring
        scored_careers = score_careers_for_student(student_profile)
        career_recommendations = call_gemini_for_careers(student_profile, scored_careers)
        
        # Step 3: Get college recommendations
        filtered_colleges = filter_colleges_for_careers(career_recommendations, student_profile)
        college_recommendations = call_gemini_for_colleges(career_recommendations, filtered_colleges, student_profile)
        
        # Step 4: Prepare response
        response = RecommendationResponse(
            student_id=payload.student_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            careers=[CareerRecommendation(**career) for career in career_recommendations[:5]],
            colleges=[CollegeRecommendation(**college) for college in college_recommendations[:5]]
        )
        
        logger.info(f"Successfully processed request for student: {payload.student_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/feedback")
async def collect_feedback(feedback: FeedbackRequest):
    """Collect student feedback for continuous improvement"""
    
    feedback_entry = {
        "student_id": feedback.student_id,
        "recommended_careers": feedback.recommended_careers,
        "selected_career": feedback.selected_career,
        "recommended_colleges": feedback.recommended_colleges,
        "selected_college": feedback.selected_college,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    feedback_data.append(feedback_entry)
    logger.info(f"Feedback collected for student: {feedback.student_id}")
    
    return {"status": "feedback_stored", "message": "Thank you for your feedback!"}

@app.get("/analytics")
async def get_analytics():
    """Get analytics data for monitoring and improvement"""
    
    if not feedback_data:
        return {
            "total_feedback": 0,
            "recent_selections": [],
            "popular_careers": [],
            "popular_colleges": []
        }
    
    # Analyze feedback data
    career_selections = [f.get("selected_career") for f in feedback_data if f.get("selected_career")]
    college_selections = [f.get("selected_college") for f in feedback_data if f.get("selected_college")]
    
    # Count popular selections
    from collections import Counter
    career_counts = Counter(career_selections)
    college_counts = Counter(college_selections)
    
    return {
        "total_feedback": len(feedback_data),
        "recent_selections": feedback_data[-10:],
        "popular_careers": career_counts.most_common(5),
        "popular_colleges": college_counts.most_common(5),
        "feedback_summary": {
            "career_selection_rate": len(career_selections) / len(feedback_data) if feedback_data else 0,
            "college_selection_rate": len(college_selections) / len(feedback_data) if feedback_data else 0
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "careers_loaded": len(careers_df) if careers_df is not None else 0,
        "colleges_loaded": len(colleges_df) if colleges_df is not None else 0,
        "feedback_entries": len(feedback_data),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)