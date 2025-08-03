"""
Query Parser Module
Parses natural language queries and         # Age patterns - More flexible matching
        self.age_patterns = [
            r'(\d{1,3})\s*(?:year|yr|y)(?:ear)?s?\s*old',
            r'(\d{1,3})\s*yo',
            r'age\s*:?\s*(\d{1,3})',
            r'(\d{1,3})\s*[MF]',  # Age followed by gender
            r'(\d{1,3})M|(\d{1,3})F',  # Common medical notation
            r'(?:^|\s+)(\d{1,3})(?:\s*(?:M|F|male|female))?(?:\s+|$)',  # Standalone number with optional gender
            r'(?:^|\s+)(\d{1,2})(?:\s+|$)'  # Just the number if between 0-99
        ]ts structured information
for better document retrieval and decision making
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Structured representation of a parsed query"""
    original_query: str
    entities: Dict[str, Any]
    intent: str
    keywords: List[str]
    enhanced_query: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class QueryParser:
    """
    Parses natural language queries to extract structured information
    such as demographics, medical procedures, locations, policy details, etc.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize query parser
        
        Args:
            spacy_model: SpaCy model name for NLP processing
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self._initialize_nlp()
        
        # Define entity patterns and mappings
        self._setup_patterns()
        
    def _initialize_nlp(self):
        """Initialize SpaCy NLP model"""
        try:
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded SpaCy model: {self.spacy_model}")
        except OSError:
            logger.warning(f"SpaCy model {self.spacy_model} not found. Using basic parsing.")
            self.nlp = None
    
    def _setup_patterns(self):
        """Setup regex patterns and keyword mappings for entity extraction"""
        
        # Age patterns - More flexible matching
        self.age_patterns = [
            r'(\d{1,3})\s*(?:year|yr|y)(?:ear)?s?\s*old',
            r'(\d{1,3})\s*yo',
            r'age\s*:?\s*(\d{1,3})',
            r'(\d{1,3})\s*[MF]',  # Age followed by gender
            r'(\d{1,3})M|(\d{1,3})F',  # Common medical notation
            r'(?:^|\s+)(\d{1,3})(?:\s*(?:M|F|male|female))?(?:\s+|$)',  # Standalone number with optional gender
            r'(?:^|\s+)(\d{1,2})(?:\s+|$)'  # Just the number if between 0-99
        ]
        
        # Gender patterns - More flexible
        self.gender_patterns = [
            r'\b(male|female|man|woman|boy|girl|M|F)\b',
            r'\b(\d+)\s*(M|F)\b',  # Age-gender combo
        ]
        
        # Medical procedure patterns - Expanded and more specific
        self.procedure_patterns = [
            r'\b(knee|hip|heart|cardiac|brain|liver|kidney|lung|spine|shoulder|ankle|wrist)\s*(surgery|operation|replacement|repair|transplant|procedure|treatment)',
            r'\b(appendectomy|cholecystectomy|bypass|angioplasty|stent|biopsy|endoscopy|colonoscopy|dialysis|chemotherapy|radiotherapy|physiotherapy|rehabilitation)\b',
            r'\b(cataract|laser|arthroscopy|laparoscopy|mastectomy|hysterectomy|tonsillectomy|dental|maternity|orthopedic|neurological|oncological)\s*(treatment|procedure|care|claim|surgery)',
            r'\b(delivery|childbirth|pregnancy|caesarean)\b', # Maternity
            r'\b(root\s*canal|filling|extraction|braces)\b', # Dental
        ]
        
        # Location patterns (Indian cities focus) - Expanded list and more flexible
        self.location_patterns = [
            r'\b(Mumbai|Delhi|Bangalore|Bengaluru|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Pimpri|Patna|Vadodara|Ghaziabad|Ludhiana|Agra|Nashik|Faridabad|Meerut|Rajkot|Kalyan|Vasai|Varanasi|Srinagar|Aurangabad|Dhanbad|Amritsar|Navi Mumbai|Allahabad|Ranchi|Howrah|Coimbatore|Jabalpur|Gwalior|Vijayawada|Jodhpur|Madurai|Raipur|Kota|Guwahati|Chandigarh|Solapur|Hubli|Bareilly|Moradabad|Mysore|Gurgaon|Aligarh|Jalandhar|Tiruchirappalli|Bhubaneswar|Salem|Warangal|Guntur|Bhiwandi|Saharanpur|Gorakhpur|Bikaner|Amravati|Noida|Jamshedpur|Bhilai|Cuttack|Firozabad|Kochi|Dehradun|Durgapur|Pondicherry|Siliguri|Jammu|Sangli|Ulhasnagar|Jalgaon|Korba|Mangalore|Erode|Belgaum|Ambattur|Tirunelveli|Malegaon|Jamnagar|Nanded|Kollam|Akola|Gulbarga|Ajmer|Thrissur|Udaipur|Asansol|Loni|Jhansi|Nellore|Mathura|Imphal|Haridwar)\b',
            r'\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Generic location after "in", "at", "from"
        ]
        
        # Policy duration patterns - More variations
        self.policy_duration_patterns = [
            r'(\d+)\s*(?:month|mon|m)(?:s)?\s*(?:old|existing|current)?\s*(?:policy|insurance|plan)',
            r'(\d+)\s*(?:year|yr|y)(?:s)?\s*(?:old|existing|current)?\s*(?:policy|insurance|plan)',
            r'(?:policy|insurance|plan)\s*(?:of|for|since)?\s*(\d+)\s*(?:month|year|mon|yr|m|y)s?',
            r'(\d+)\s*day(?:s)?\s*(?:old|existing|current)?\s*(?:policy|insurance|plan)',
        ]
        
        # Insurance-related keywords - Expanded
        self.insurance_keywords = [
            'claim', 'coverage', 'policy', 'premium', 'deductible', 'copay',
            'covered', 'excluded', 'benefit', 'reimbursement', 'approval',
            'pre-existing', 'waiting period', 'cashless', 'network hospital', 'sum assured',
            'renewal', 'endorsement', 'nominee', 'insurer', 'insured', 'hospitalization',
            'OPD', 'IPD', 'critical illness', 'accident', 'disease', 'illness', 'diagnosis'
        ]
        
        # Medical keywords - Expanded
        self.medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'prescription', 'doctor',
            'hospital', 'clinic', 'emergency', 'consultation', 'therapy',
            'rehabilitation', 'follow-up', 'complications', 'recovery', 'ICU', 'room rent',
            'patient', 'medical report', 'discharge summary', 'ailment', 'symptoms', 'health'
        ]
        
        # Intent patterns - More robust
        self.intent_patterns = {
            'claim_inquiry': [
                r'\b(claim|coverage|covered|eligible|approve|reimburse|settlement|pay)\b',
                r'\b(can\s+I|will\s+you|is\s+this|am\s+I)\b.*\b(covered|eligible|approved|reimbursed)\b',
                r'\b(how\s+to\s+claim|process\s+claim|claim\s+procedure)\b'
            ],
            'policy_check': [
                r'\b(policy|insurance|plan)\b.*\b(details|information|status|terms|conditions|document)\b',
                r'\b(what|which|how)\b.*\b(policy|coverage|benefit|plan)\b',
                r'\b(check\s+my\s+policy|policy\s+status)\b'
            ],
            'cost_inquiry': [
                r'\b(cost|price|amount|fee|charge|bill|expense)\b',
                r'\b(how\s+much|what.*cost|price.*for|estimate)\b'
            ],
            'eligibility_check': [
                r'\b(eligible|qualify|entitled|allowed|can\s+get)\b',
                r'\b(can\s+I|am\s+I\s+able|is\s+it\s+possible)\b',
                r'\b(eligibility\s+criteria|who\s+is\s+eligible)\b'
            ],
            'document_request': [
                r'\b(need|get|provide|send)\s+me\s+(?:the)?\s*(?:policy)?\s*(document|copy|paperwork|form)\b',
                r'\b(where\s+can\s+I\s+find|access\s+my)\s+policy\b'
            ]
        }
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query and extract structured information
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted information
        """
        logger.info(f"Parsing query: {query}")
        
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Try multiple variations of the query for better entity extraction
        variations = [
            cleaned_query,
            # Add common variations
            cleaned_query.replace(",", " "),  # Handle comma-separated format
            cleaned_query.replace("-", " "),  # Handle hyphenated format
            # Add expanded forms
            cleaned_query.replace("m", " male ").replace("f", " female "),
            cleaned_query.replace("yr", " year ").replace("y", " year "),
            cleaned_query.replace("mon", " month ").replace("m", " month ")
        ]
        
        # Extract entities from all variations
        all_entities = {}
        for variation in variations:
            entities = self._extract_entities(variation)
            # Merge new entities
            for key, value in entities.items():
                if key not in all_entities or not all_entities[key]:
                    all_entities[key] = value
        
        # Determine intent - check multiple variations for better accuracy
        intents = [self._determine_intent(var) for var in variations]
        intent = max(set(intents), key=intents.count)  # Use most common intent
        
        # Extract keywords from all variations
        all_keywords = set()
        for variation in variations:
            all_keywords.update(self._extract_keywords(variation))
        keywords = list(all_keywords)
        
        # Create enhanced query using all extracted information
        enhanced_query = self._enhance_query(cleaned_query, all_entities, keywords)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(all_entities, intent, keywords)
        
        parsed_query = ParsedQuery(
            original_query=query,
            entities=all_entities,
            intent=intent,
            keywords=keywords,
            enhanced_query=enhanced_query,
            confidence=confidence,
            metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'query_length': len(query),
                'entities_found': len(all_entities),
                'keywords_found': len(keywords)
            }
        )
        
        logger.info(f"Parsed query with confidence {confidence:.2f}")
        return parsed_query
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query text"""
        # Convert to lowercase for processing
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep essential punctuation
        cleaned = re.sub(r'[^\w\s\-,.]', '', cleaned)
        
        return cleaned
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from the query"""
        entities = {}
        
        # Extract age
        age = self._extract_age(query)
        if age:
            entities['age'] = age
        
        # Extract gender
        gender = self._extract_gender(query)
        if gender:
            entities['gender'] = gender
        
        # Extract medical procedures
        procedures = self._extract_procedures(query)
        if procedures:
            entities['procedures'] = procedures
        
        # Extract locations
        locations = self._extract_locations(query)
        if locations:
            entities['locations'] = locations
        
        # Extract policy duration
        policy_duration = self._extract_policy_duration(query)
        if policy_duration:
            entities['policy_duration'] = policy_duration
        
        # Extract amounts/numbers
        amounts = self._extract_amounts(query)
        if amounts:
            entities['amounts'] = amounts
        
        # Use SpaCy for additional entity extraction if available
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(query)
            entities.update(spacy_entities)
        
        return entities
    
    def _extract_age(self, query: str) -> Optional[int]:
        """Extract age from query"""
        for pattern in self.age_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Find the first non-None group
                for group in match.groups():
                    if group and group.isdigit():
                        age = int(group)
                        if 0 <= age <= 120:  # Reasonable age range
                            return age
        return None
    
    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract gender from query"""
        for pattern in self.gender_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                gender_text = match.group(1).lower()
                if gender_text in ['male', 'man', 'boy', 'm']:
                    return 'male'
                elif gender_text in ['female', 'woman', 'girl', 'f']:
                    return 'female'
        return None
    
    def _extract_procedures(self, query: str) -> List[str]:
        """Extract medical procedures from query"""
        procedures = []
        for pattern in self.procedure_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    procedures.extend([m for m in match if m])
                else:
                    procedures.append(match)
        
        # Remove duplicates and clean
        procedures = list(set([p.strip().lower() for p in procedures if p.strip()]))
        return procedures
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract locations from query"""
        locations = []
        for pattern in self.location_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    locations.extend([m for m in match if m])
                else:
                    locations.append(match)
        
        # Clean and deduplicate
        locations = list(set([loc.strip().title() for loc in locations if loc.strip()]))
        return locations
    
    def _extract_policy_duration(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract policy duration information"""
        for pattern in self.policy_duration_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                duration_value = int(match.group(1))
                duration_text = match.group(0).lower()
                
                if 'month' in duration_text or 'mon' in duration_text or duration_text.endswith('m'):
                    return {'value': duration_value, 'unit': 'months'}
                elif 'year' in duration_text or 'yr' in duration_text or duration_text.endswith('y'):
                    return {'value': duration_value, 'unit': 'years'}
        
        return None
    
    def _extract_amounts(self, query: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts from query"""
        amounts = []
        
        # Pattern for currency amounts
        currency_patterns = [
            r'(?:rs\.?|rupees?|inr)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|rupees?|inr)',
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd)',
        ]
        
        for pattern in currency_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    currency = 'INR'  # Default to INR
                    if '$' in match.group(0) or 'dollar' in match.group(0).lower() or 'usd' in match.group(0).lower():
                        currency = 'USD'
                    
                    amounts.append({
                        'value': amount,
                        'currency': currency,
                        'original_text': match.group(0)
                    })
                except ValueError:
                    continue
        
        return amounts
    
    def _extract_spacy_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities using SpaCy NER"""
        entities = {}
        
        try:
            doc = self.nlp(query)
            
            spacy_entities = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],  # Geopolitical entities (cities, countries)
                'DATE': [],
                'MONEY': [],
                'QUANTITY': []
            }
            
            for ent in doc.ents:
                if ent.label_ in spacy_entities:
                    spacy_entities[ent.label_].append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': ent._.confidence if hasattr(ent._, 'confidence') else 1.0
                    })
            
            # Only add non-empty entity types
            for entity_type, entity_list in spacy_entities.items():
                if entity_list:
                    entities[f'spacy_{entity_type.lower()}'] = entity_list
            
        except Exception as e:
            logger.warning(f"SpaCy entity extraction failed: {str(e)}")
        
        return entities
    
    def _determine_intent(self, query: str) -> str:
        """Determine the main intent of the query"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general_inquiry'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query using a more robust method."""
        keywords = set()

        # Add insurance-related keywords
        for keyword in self.insurance_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                keywords.add(keyword)

        # Add medical keywords
        for keyword in self.medical_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                keywords.add(keyword)

        # Use SpaCy for more intelligent keyword extraction
        if self.nlp:
            try:
                doc = self.nlp(query)
                for token in doc:
                    # Extract nouns, proper nouns, and adjectives as keywords
                    if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 2:
                        keywords.add(token.lemma_.lower())
            except Exception as e:
                logger.warning(f"SpaCy keyword extraction failed: {str(e)}")
        
        return list(keywords)

    def _enhance_query(self, query: str, entities: Dict[str, Any], keywords: List[str]) -> str:
        """Create a structured and enhanced query for better retrieval."""
        # Start with the core intent and procedure
        intent = self._determine_intent(query)
        enhanced_parts = [intent]

        if 'procedures' in entities:
            enhanced_parts.extend(entities['procedures'])
        
        # Add other entities as key-value pairs for clarity
        for entity, value in entities.items():
            if entity != 'procedures':
                if isinstance(value, dict):
                    enhanced_parts.append(f"{entity}: {value.get('value')} {value.get('unit', '')}".strip())
                elif isinstance(value, list):
                    # Ensure list items are properly handled, especially for spacy entities
                    formatted_list = []
                    for item in value:
                        if isinstance(item, dict) and 'text' in item:
                            formatted_list.append(item['text'])
                        else:
                            formatted_list.append(str(item))
                    enhanced_parts.append(f"{entity}: {', '.join(formatted_list)}")
                else:
                    enhanced_parts.append(f"{entity}: {value}")

        # Add a few highly relevant keywords
        enhanced_parts.extend(keywords[:5]) # Increased to top 5 keywords

        return " | ".join(enhanced_parts)
    
    def _calculate_confidence(self, entities: Dict[str, Any], intent: str, keywords: List[str]) -> float:
        """Calculate confidence score for the parsed query"""
        confidence = 0.0
        
        # Base confidence - higher base for queries with key information
        has_key_info = any([entities.get('age'), entities.get('procedures'), entities.get('policy_duration'), entities.get('amounts')])
        confidence += 0.3 if has_key_info else 0.1
        
        # Entity extraction confidence - weighted by importance
        if entities:
            entity_weights = {
                'age': 0.15,
                'gender': 0.05,
                'procedures': 0.2,
                'locations': 0.1,
                'policy_duration': 0.15,
                'amounts': 0.1,
                'spacy_person': 0.05,
                'spacy_org': 0.05,
                'spacy_gpe': 0.05,
                'spacy_date': 0.05,
                'spacy_money': 0.05,
                'spacy_quantity': 0.05
            }
            
            for entity_type, weight in entity_weights.items():
                if entities.get(entity_type):
                    confidence += weight
        
        # Intent detection confidence - higher weight for specific intents
        intent_weights = {
            'claim_inquiry': 0.2,
            'policy_check': 0.15,
            'cost_inquiry': 0.15,
            'eligibility_check': 0.15,
            'document_request': 0.1,
            'general_inquiry': 0.05
        }
        confidence += intent_weights.get(intent, 0.05)
        
        # Keyword relevance confidence - weighted by relevance
        if keywords:
            relevant_keywords = ['surgery', 'treatment', 'policy', 'insurance', 'hospital', 'claim', 'covered', 'eligible', 'cost', 'amount']
            relevant_count = sum(1 for k in keywords if any(r in k.lower() for r in relevant_keywords))
            confidence += min(0.2, relevant_count * 0.02) # Max 0.2 for keywords
        
        return min(1.0, confidence)
    
    def get_query_stats(self, parsed_queries: List[ParsedQuery]) -> Dict[str, Any]:
        """Get statistics about parsed queries"""
        if not parsed_queries:
            return {"total_queries": 0}
        
        intents = {}
        avg_confidence = sum(q.confidence for q in parsed_queries) / len(parsed_queries)
        
        for query in parsed_queries:
            intent = query.intent
            intents[intent] = intents.get(intent, 0) + 1
        
        return {
            "total_queries": len(parsed_queries),
            "average_confidence": avg_confidence,
            "intent_distribution": intents,
            "avg_entities_per_query": sum(len(q.entities) for q in parsed_queries) / len(parsed_queries),
            "avg_keywords_per_query": sum(len(q.keywords) for q in parsed_queries) / len(parsed_queries)
        }

# Example usage and testing
# Example usage and testing
if __name__ == "__main__":
    parser = QueryParser()

    # Test queries
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "46M, knee surgery, Pune, 3-month policy",
        "25 year old female wants to know if her heart surgery is covered",
        "I am 35 years old and need cataract operation in Mumbai with an existing 2 year policy",
        "Is angioplasty covered in my 1-year-old insurance policy for a 60M patient in Delhi?",
        "What’s the cost of a liver transplant in Hyderabad with 6 months insurance?",
        "Can a 70-year-old female get approval for a hip replacement in Chennai?",
        "Need to check if pre-existing diabetes treatment is eligible under my 2-year policy",
        "Rs 2,50,000 reimbursement request for a brain surgery in Bangalore for 45M",
        "I want to claim for shoulder arthroscopy done last month in Jaipur"
    ]

    parsed_results = []

    for query in test_queries:
        result = parser.parse_query(query)
        print("\n--- Parsed Query ---")
        print(f"Original: {result.original_query}")
        print(f"Intent: {result.intent}")
        print(f"Entities: {result.entities}")
        print(f"Keywords: {result.keywords}")
        print(f"Enhanced Query: {result.enhanced_query}")
        print(f"Confidence Score: {result.confidence:.2f}")
        parsed_results.append(result)

    # Show query parsing statistics
    stats = parser.get_query_stats(parsed_results)
    print("\n=== Query Parsing Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
