"""
Query Parser Module
Parses natural language queries and extracts structured information
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
        
        # Age patterns
        self.age_patterns = [
            r'(\d{1,3})\s*(?:year|yr|y)(?:ear)?s?\s*old',
            r'(\d{1,3})\s*yo',
            r'age\s*:?\s*(\d{1,3})',
            r'(\d{1,3})\s*[MF]',  # Age followed by gender
            r'(\d{1,3})M|(\d{1,3})F',  # Common medical notation
        ]
        
        # Gender patterns
        self.gender_patterns = [
            r'\b(male|female|man|woman|boy|girl|M|F)\b',
            r'\b(\d+)\s*(M|F)\b',  # Age-gender combo
        ]
        
        # Medical procedure patterns
        self.procedure_patterns = [
            r'\b(surgery|operation|procedure|treatment)\b',
            r'\b(knee|hip|heart|brain|liver|kidney|lung|spine|shoulder|ankle|wrist)\s*(surgery|operation|replacement|repair)',
            r'\b(appendectomy|cholecystectomy|bypass|angioplasty|stent|biopsy|endoscopy|colonoscopy)\b',
            r'\b(cataract|laser|arthroscopy|laparoscopy|minimally\s*invasive)\b',
        ]
        
        # Location patterns (Indian cities focus)
        self.location_patterns = [
            r'\b(Mumbai|Delhi|Bangalore|Bengaluru|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Pimpri|Patna|Vadodara|Ghaziabad|Ludhiana|Agra|Nashik|Faridabad|Meerut|Rajkot|Kalyan|Vasai|Varanasi|Srinagar|Aurangabad|Dhanbad|Amritsar|Navi Mumbai|Allahabad|Ranchi|Howrah|Coimbatore|Jabalpur|Gwalior|Vijayawada|Jodhpur|Madurai|Raipur|Kota|Guwahati|Chandigarh|Solapur|Hubli|Bareilly|Moradabad|Mysore|Gurgaon|Aligarh|Jalandhar|Tiruchirappalli|Bhubaneswar|Salem|Warangal|Guntur|Bhiwandi|Saharanpur|Gorakhpur|Bikaner|Amravati|Noida|Jamshedpur|Bhilai|Cuttack|Firozabad|Kochi|Dehradun|Durgapur|Pondicherry|Siliguri|Jammu|Sangli|Ulhasnagar|Jalgaon|Korba|Mangalore|Erode|Belgaum|Ambattur|Tirunelveli|Malegaon|Jamnagar|Nanded|Kollam|Akola|Gulbarga|Ajmer|Thrissur|Udaipur|Asansol|Loni|Jhansi|Nellore|Mathura|Imphal|Haridwar)\b',
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Generic location after "in"
        ]
        
        # Policy duration patterns
        self.policy_duration_patterns = [
            r'(\d+)\s*(?:month|mon|m)(?:old|s)?\s*(?:policy|insurance)',
            r'(\d+)\s*(?:year|yr|y)(?:old|s)?\s*(?:policy|insurance)',
            r'(?:policy|insurance)\s*(?:of|for|since)?\s*(\d+)\s*(?:month|year|mon|yr|m|y)s?',
            r'(\d+)\s*(?:month|year|mon|yr|m|y)s?\s*(?:old|existing|current)\s*(?:policy|insurance)',
        ]
        
        # Insurance-related keywords
        self.insurance_keywords = [
            'claim', 'coverage', 'policy', 'premium', 'deductible', 'copay',
            'covered', 'excluded', 'benefit', 'reimbursement', 'approval',
            'pre-existing', 'waiting period', 'cashless', 'network hospital'
        ]
        
        # Medical keywords
        self.medical_keywords = [
            'diagnosis', 'treatment', 'medication', 'prescription', 'doctor',
            'hospital', 'clinic', 'emergency', 'consultation', 'therapy',
            'rehabilitation', 'follow-up', 'complications', 'recovery'
        ]
        
        # Intent patterns
        self.intent_patterns = {
            'claim_inquiry': [
                r'\b(claim|coverage|covered|eligible|approve|reimburse)\b',
                r'\b(can\s+I|will\s+you|is\s+this|am\s+I)\b.*\b(covered|eligible|approved)\b'
            ],
            'policy_check': [
                r'\b(policy|insurance|plan)\b.*\b(details|information|status)\b',
                r'\b(what|which|how)\b.*\b(policy|coverage|benefit)\b'
            ],
            'cost_inquiry': [
                r'\b(cost|price|amount|fee|charge|bill)\b',
                r'\b(how\s+much|what.*cost|price.*for)\b'
            ],
            'eligibility_check': [
                r'\b(eligible|qualify|entitled|allowed)\b',
                r'\b(can\s+I|am\s+I\s+able|is\s+it\s+possible)\b'
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
        
        # Extract entities
        entities = self._extract_entities(cleaned_query)
        
        # Determine intent
        intent = self._determine_intent(cleaned_query)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_query)
        
        # Create enhanced query
        enhanced_query = self._enhance_query(cleaned_query, entities, keywords)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(entities, intent, keywords)
        
        parsed_query = ParsedQuery(
            original_query=query,
            entities=entities,
            intent=intent,
            keywords=keywords,
            enhanced_query=enhanced_query,
            confidence=confidence,
            metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'query_length': len(query),
                'entities_found': len(entities),
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
        """Extract relevant keywords from the query"""
        keywords = []
        
        # Add insurance-related keywords
        for keyword in self.insurance_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                keywords.append(keyword)
        
        # Add medical keywords
        for keyword in self.medical_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                keywords.append(keyword)
        
        # Extract important nouns using SpaCy if available
        if self.nlp:
            try:
                doc = self.nlp(query)
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        len(token.text) > 3 and 
                        token.text.lower() not in keywords):
                        keywords.append(token.text.lower())
            except Exception as e:
                logger.warning(f"SpaCy keyword extraction failed: {str(e)}")
        
        return list(set(keywords))  # Remove duplicates
    
    def _enhance_query(self, query: str, entities: Dict[str, Any], keywords: List[str]) -> str:
        """Create an enhanced query for better retrieval"""
        enhanced_parts = [query]
        
        # Add entity-based enhancements
        if 'age' in entities and 'gender' in entities:
            enhanced_parts.append(f"{entities['age']} year old {entities['gender']}")
        
        if 'procedures' in entities:
            enhanced_parts.extend(entities['procedures'])
        
        if 'locations' in entities:
            enhanced_parts.extend(entities['locations'])
        
        if 'policy_duration' in entities:
            duration = entities['policy_duration']
            enhanced_parts.append(f"{duration['value']} {duration['unit']} policy")
        
        # Add relevant keywords
        enhanced_parts.extend(keywords[:5])  # Top 5 keywords
        
        return ' '.join(enhanced_parts)
    
    def _calculate_confidence(self, entities: Dict[str, Any], intent: str, keywords: List[str]) -> float:
        """Calculate confidence score for the parsed query"""
        confidence = 0.0
        
        # Base confidence
        confidence += 0.3
        
        # Entity extraction confidence
        if entities:
            confidence += min(0.4, len(entities) * 0.1)
        
        # Intent detection confidence
        if intent != 'general_inquiry':
            confidence += 0.2
        
        # Keyword relevance confidence
        if keywords:
            confidence += min(0.1, len(keywords) * 0.02)
        
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
