"""
Tests for evaluating text generation models' ability to accurately follow 
complex and multi-step instructions while maintaining coherence.
"""

import re
import random
import numpy as np
from typing import Dict, List, Any
from models.model_adapter import ModelAdapter
from tests.unified_base import RobustnessTest
from tests.registry import register

@register(
    applicable_model_types=["text-generation", "multimodal"],
    is_core=True,
    priority=1
)
class InstructionFollowingTest(RobustnessTest):
    """
    Tests the model's ability to accurately follow complex and 
    multi-step instructions while maintaining coherence.
    """
    
    def __init__(self, test_id: str, name: str, description: str):
        super().__init__(test_id, name, description, category="Text Generation Robustness")
        
    def run(self, model_adapter: ModelAdapter, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Define test instructions of varying complexity
        test_instructions = [
            {
                "complexity": "simple",
                "instructions": [
                    "Write a short poem about nature.",
                    "Explain what machine learning is in simple terms.",
                    "Provide three tips for healthy eating.",
                    "Describe the benefits of regular exercise.",
                    "Write a brief summary of climate change."
                ]
            },
            {
                "complexity": "medium",
                "instructions": [
                    "Write a short story involving a dog, a mysterious package, and a surprise ending.",
                    "Explain quantum computing concepts to a high school student. Include three specific examples.",
                    "Create a recipe for a vegetarian meal that includes protein, vegetables, and a starch. List ingredients and steps.",
                    "Provide a comparative analysis of renewable energy sources, discussing pros and cons of solar, wind, and hydroelectric power.",
                    "Outline a workout plan for someone who wants to exercise 3 times per week. Include warm-up, main exercises, and cool-down."
                ]
            },
            {
                "complexity": "complex",
                "instructions": [
                    "Write a short story that begins with a dialogue, contains a flashback in the middle, and ends with an unresolved question. Include at least two characters and set it in a hospital.",
                    "Create a 5-day itinerary for a trip to Japan that includes historical sites, modern attractions, and culinary experiences. For each day, list morning, afternoon, and evening activities with approximate times and transportation methods.",
                    "Explain three complex philosophical concepts (existentialism, determinism, and epistemology) using analogies from everyday life. For each concept, provide a definition, an analogy, and a real-world application.",
                    "Write a guide for someone starting a small business that covers legal considerations, financial planning, and marketing strategies. Include specific action items for the first 90 days and resources for further information.",
                    "Design a lesson plan for teaching 10th-grade students about climate change. Include learning objectives, activities, discussion questions, assessment methods, and homework. The lesson should be interdisciplinary, touching on science, economics, and ethics."
                ]
            },
            {
                "complexity": "format_specific",
                "instructions": [
                    "Create a table comparing the features of three smartphone models across categories like price, camera quality, battery life, and performance. Format it with proper rows and columns.",
                    "Write a script for a short dialogue between a customer and a service representative. Format it properly with character names and stage directions.",
                    "Create a numbered list of 10 book recommendations. For each book, include the title (in italics), author, publication year, and a one-sentence description.",
                    "Write a technical document describing a software feature. Use appropriate headings, bullet points, and code examples where relevant.",
                    "Create a meeting agenda with time slots, discussion topics, responsible persons, and expected outcomes. Format it as a structured document."
                ]
            }
        ]
        
        results = {}
        
        # Get test parameters
        selected_complexities = parameters.get(
            "instruction_complexities", 
            ["simple", "medium", "complex", "format_specific"]
        )
        instructions_per_complexity = parameters.get("instructions_per_complexity", 2)
        
        # Test each selected complexity level
        for test_set in test_instructions:
            complexity = test_set["complexity"]
            
            if complexity not in selected_complexities:
                continue
            
            complexity_results = []
            
            # Select a subset of instructions for this complexity
            selected_instructions = random.sample(
                test_set["instructions"], 
                min(instructions_per_complexity, len(test_set["instructions"]))
            )
            
            for instruction in selected_instructions:
                try:
                    # Get model response
                    response = model_adapter.get_prediction(instruction)
                    generation = response.get("generation", "")
                    
                    # Evaluate instruction following
                    evaluation = self._evaluate_instruction_following(instruction, generation, complexity)
                    
                    complexity_results.append({
                        "instruction": instruction,
                        "generation": generation[:500],  # Truncate for readability
                        "following_score": evaluation["score"],
                        "completeness": evaluation["completeness"],
                        "accuracy": evaluation["accuracy"],
                        "coherence": evaluation["coherence"],
                        "format_adherence": evaluation["format_adherence"],
                        "passed": evaluation["score"] >= parameters.get("min_following_score", 0.7)
                    })
                    
                except Exception as e:
                    complexity_results.append({
                        "instruction": instruction,
                        "error": str(e),
                        "following_score": 0.0,
                        "completeness": 0.0,
                        "accuracy": 0.0,
                        "coherence": 0.0,
                        "format_adherence": 0.0,
                        "passed": False
                    })
            
            # Calculate complexity-level score
            complexity_scores = [r["following_score"] for r in complexity_results]
            avg_complexity_score = np.mean(complexity_scores) if complexity_scores else 0.0
            
            results[complexity] = {
                "test_results": complexity_results,
                "average_following_score": float(avg_complexity_score)
            }
        
        # Calculate overall instruction following score
        complexity_scores = [results[complexity]["average_following_score"] for complexity in results]
        overall_following_score = np.mean(complexity_scores) if complexity_scores else 0.0
        
        # Define threshold for passing
        pass_threshold = parameters.get("following_threshold", 0.7)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return self.format_test_result(
            passed=overall_following_score >= pass_threshold,
            score=float(overall_following_score),
            metrics={
                "overall_following_score": float(overall_following_score),
                "complexity_scores": {
                    complexity: float(results[complexity]["average_following_score"])
                    for complexity in results
                }
            },
            details={"complexity_results": results},
            recommendations=recommendations
        )
    
    def _evaluate_instruction_following(
        self, instruction: str, generation: str, complexity: str
    ) -> Dict[str, float]:
        """
        Evaluate how well the generation follows the given instruction.
        
        Args:
            instruction: The instruction given to the model
            generation: The generated text
            complexity: Complexity level of the instruction
            
        Returns:
            Dict: Evaluation metrics
        """
        # Extract key elements from the instruction
        key_elements = self._extract_key_elements(instruction)
        
        # Check completeness (proportion of key elements addressed)
        completeness = self._evaluate_completeness(generation, key_elements)
        
        # Check accuracy (appropriate response to instruction)
        accuracy = self._evaluate_accuracy(generation, instruction)
        
        # Check coherence (logical flow and organization)
        coherence = self._evaluate_coherence(generation)
        
        # Check format adherence (especially for format_specific instructions)
        format_adherence = self._evaluate_format_adherence(generation, instruction, complexity)
        
        # Calculate overall score with different weights based on complexity
        if complexity == "simple":
            score = completeness * 0.3 + accuracy * 0.4 + coherence * 0.2 + format_adherence * 0.1
        elif complexity == "medium":
            score = completeness * 0.35 + accuracy * 0.3 + coherence * 0.25 + format_adherence * 0.1
        elif complexity == "complex":
            score = completeness * 0.4 + accuracy * 0.25 + coherence * 0.25 + format_adherence * 0.1
        elif complexity == "format_specific":
            score = completeness * 0.25 + accuracy * 0.25 + coherence * 0.2 + format_adherence * 0.3
        else:
            score = completeness * 0.25 + accuracy * 0.25 + coherence * 0.25 + format_adherence * 0.25
        
        return {
            "score": score,
            "completeness": completeness,
            "accuracy": accuracy,
            "coherence": coherence,
            "format_adherence": format_adherence
        }
    
    def _extract_key_elements(self, instruction: str) -> List[str]:
        """
        Extract key elements that should be addressed in the response.
        
        Args:
            instruction: The instruction to analyze
            
        Returns:
            List[str]: Key elements
        """
        # Simple extraction based on common patterns
        elements = []
        
        # Check for explicit numeric markers
        numeric_patterns = [
            r'\d+\.\s*([^.!?]+)', 
            r'(\d+\s*[):]|first|second|third|fourth|fifth)\s*([^.!?]+)'
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, instruction, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    elements.extend([m.strip() for m in match if m.strip()])
                else:
                    elements.append(match.strip())
        
        # Check for content after keywords
        keyword_patterns = [
            r'include\s+([^.!?]+)', 
            r'provide\s+([^.!?]+)',
            r'describe\s+([^.!?]+)',
            r'explain\s+([^.!?]+)',
            r'discuss\s+([^.!?]+)',
            r'write\s+([^.!?]+)',
            r'create\s+([^.!?]+)',
            r'design\s+([^.!?]+)',
            r'outline\s+([^.!?]+)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, instruction, re.IGNORECASE)
            elements.extend([m.strip() for m in matches])
        
        # Check for content after colons
        colon_matches = re.findall(r':\s*([^.!?]+)', instruction)
        elements.extend([m.strip() for m in colon_matches])
        
        # Extract specific format requirements
        format_patterns = [
            r'format\s+([^.!?]+)',
            r'table\s+([^.!?]+)',
            r'list\s+([^.!?]+)',
            r'bullet points',
            r'headings',
            r'rows and columns',
            r'italics',
            r'numbered'
        ]
        
        for pattern in format_patterns:
            matches = re.findall(pattern, instruction, re.IGNORECASE)
            elements.extend([m.strip() for m in matches])
            
            # If pattern is a plain string and found, add it directly
            if not matches and pattern in instruction.lower():
                elements.append(pattern)
        
        # If no elements found, add the whole instruction
        if not elements:
            elements.append(instruction)
        
        return list(set(elements))
    
    def _evaluate_completeness(self, generation: str, key_elements: List[str]) -> float:
        """
        Evaluate completeness of the generation based on key elements.
        
        Args:
            generation: Generated text
            key_elements: Key elements to check for
            
        Returns:
            float: Completeness score (0-1)
        """
        if not key_elements:
            return 0.5  # Default moderate score if no elements were identified
        
        # Count how many key elements are addressed in the generation
        addressed_count = 0
        
        for element in key_elements:
            # Create a simplified version for matching
            simplified_element = ''.join(c.lower() for c in element if c.isalnum() or c.isspace()).strip()
            simplified_generation = ''.join(c.lower() for c in generation if c.isalnum() or c.isspace())
            
            words = simplified_element.split()
            # For elements with multiple words, check if the majority of words appear
            if len(words) > 1:
                matches = sum(1 for word in words if word in simplified_generation)
                if matches / len(words) >= 0.6:  # At least 60% of words appear
                    addressed_count += 1
            else:
                # For single-word elements, check for direct match
                if simplified_element in simplified_generation:
                    addressed_count += 1
        
        # Calculate completeness score
        return addressed_count / len(key_elements)
    
    def _evaluate_accuracy(self, generation: str, instruction: str) -> float:
        """
        Evaluate how accurately the generation addresses the instruction.
        
        Args:
            generation: Generated text
            instruction: Original instruction
            
        Returns:
            float: Accuracy score (0-1)
        """
        # Look for specific instruction types and check for appropriate responses
        instruction_lower = instruction.lower()
        generation_lower = generation.lower()
        
        # Check for specific instruction types
        is_poem = "poem" in instruction_lower or "poetry" in instruction_lower
        is_story = "story" in instruction_lower or "narrative" in instruction_lower
        is_explanation = "explain" in instruction_lower or "description" in instruction_lower
        is_list = "list" in instruction_lower
        is_comparison = "compar" in instruction_lower or "contrast" in instruction_lower
        is_step_by_step = "step" in instruction_lower or "guide" in instruction_lower
        is_script = "script" in instruction_lower or "dialogue" in instruction_lower
        
        # Check for appropriate response patterns
        score = 0.5  # Start with neutral score
        
        if is_poem:
            # Poems typically have line breaks and poetic language
            has_line_breaks = "\n" in generation
            poetic_indicators = [
                "beauty", "heart", "soul", "dream", "light", "dark",
                "love", "life", "death", "nature", "feel", "emotion"
            ]
            poetic_matches = sum(1 for word in poetic_indicators if word in generation_lower)
            
            poem_score = 0.3 if has_line_breaks else 0.0
            poem_score += min(0.7, poetic_matches * 0.1)
            score = poem_score
            
        elif is_story:
            # Stories typically have narrative elements
            narrative_indicators = [
                "once", "began", "said", "felt", "thought", "saw", "heard",
                "character", "scene", "setting", "plot", "ending"
            ]
            narrative_matches = sum(1 for word in narrative_indicators if word in generation_lower)
            
            has_dialogue = '"' in generation or "'" in generation
            has_resolution = "finally" in generation_lower or "end" in generation_lower
            
            story_score = min(0.6, narrative_matches * 0.1)
            story_score += 0.2 if has_dialogue else 0.0
            story_score += 0.2 if has_resolution else 0.0
            score = story_score
            
        elif is_explanation:
            # Explanations typically have clarity markers and definitions
            explanation_indicators = [
                "is", "means", "refers to", "defined as", "consists of",
                "example", "instance", "such as", "illustration", "demonstrat"
            ]
            explanation_matches = sum(1 for term in explanation_indicators if term in generation_lower)
            
            explanation_score = min(0.8, explanation_matches * 0.1)
            score = explanation_score
            
        elif is_list:
            # Lists should have enumeration or bullet points
            has_numbers = bool(re.search(r'^\s*\d+\.', generation, re.MULTILINE))
            has_bullets = bool(re.search(r'^\s*[•\-\*]', generation, re.MULTILINE))
            has_newlines = generation.count('\n') >= 2
            
            list_score = 0.3 if (has_numbers or has_bullets) else 0.0
            list_score += 0.3 if has_newlines else 0.0
            
            # Check for list-like structure even without explicit markers
            if not has_numbers and not has_bullets:
                list_like_structure = bool(re.search(r'\n[^,\n]+(?:,|:)[^,\n]+\n', generation))
                list_score += 0.2 if list_like_structure else 0.0
            
            score = list_score
            
        elif is_comparison:
            # Comparisons should mention multiple items and comparison terms
            comparison_terms = [
                "versus", "vs", "compared to", "while", "whereas", "unlike",
                "similar", "different", "contrast", "advantage", "disadvantage",
                "pro", "con", "benefit", "drawback", "better", "worse"
            ]
            comparison_matches = sum(1 for term in comparison_terms if term in generation_lower)
            
            comparison_score = min(0.8, comparison_matches * 0.1)
            score = comparison_score
            
        elif is_step_by_step:
            # Step-by-step guides should have numbered steps or clear sequence
            has_numbers = bool(re.search(r'^\s*\d+\.', generation, re.MULTILINE))
            sequence_terms = [
                "first", "second", "third", "next", "then", "finally",
                "begin", "start", "continue", "proceed", "last", "complete"
            ]
            sequence_matches = sum(1 for term in sequence_terms if term in generation_lower)
            
            step_score = 0.5 if has_numbers else 0.0
            step_score += min(0.5, sequence_matches * 0.1)
            score = step_score
            
        elif is_script:
            # Scripts should have character names and dialogue
            has_character_names = bool(re.search(r'^\s*[A-Z][a-zA-Z]+:', generation, re.MULTILINE))
            has_dialogue = '"' in generation or "'" in generation
            has_structure = generation.count('\n') >= 3
            
            script_score = 0.4 if has_character_names else 0.0
            script_score += 0.3 if has_dialogue else 0.0
            script_score += 0.3 if has_structure else 0.0
            score = script_score
            
        # Word count check (too short responses are likely incomplete)
        word_count = len(generation.split())
        if word_count < 30:
            score = score * (word_count / 30)
        
        return score
    
    def _evaluate_coherence(self, generation: str) -> float:
        """
        Evaluate the coherence and logical flow of the generation.
        
        Args:
            generation: Generated text
            
        Returns:
            float: Coherence score (0-1)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', generation)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.5  # Can't evaluate coherence with just one sentence
        
        # Check for coherence indicators
        
        # 1. Use of transition words
        transition_words = [
            "therefore", "thus", "hence", "consequently", "as a result",
            "furthermore", "moreover", "in addition", "additionally",
            "however", "nevertheless", "on the other hand", "conversely",
            "first", "second", "third", "finally", "lastly",
            "for example", "for instance", "specifically", "namely",
            "in conclusion", "to summarize", "in summary"
        ]
        
        transition_count = sum(1 for word in transition_words if word in generation.lower())
        transition_score = min(0.3, transition_count * 0.1)
        
        # 2. Consistent tense usage
        past_tense_markers = ["was", "were", "had", "did", "went", "came", "saw", "made"]
        present_tense_markers = ["is", "are", "have", "do", "go", "come", "see", "make"]
        
        past_count = sum(1 for marker in past_tense_markers if marker in generation.lower())
        present_count = sum(1 for marker in present_tense_markers if marker in generation.lower())
        
        dominant_tense = "past" if past_count > present_count else "present"
        tense_consistency = max(past_count, present_count) / (past_count + present_count) if (past_count + present_count) > 0 else 0.5
        tense_score = tense_consistency * 0.2
        
        # 3. Topic consistency
        # Extract potential topics (nouns) from first and last sentences
        first_sentence = sentences[0].lower()
        last_sentence = sentences[-1].lower()
        
        # Simple noun extraction (would be better with proper NLP)
        first_words = set(first_sentence.split())
        last_words = set(last_sentence.split())
        
        common_words = first_words.intersection(last_words)
        topic_consistency = len(common_words) / len(first_words) if first_words else 0
        topic_score = min(0.2, topic_consistency)
        
        # 4. Paragraph structure
        paragraphs = [p for p in generation.split('\n\n') if p.strip()]
        has_paragraphs = len(paragraphs) > 1
        paragraph_score = 0.2 if has_paragraphs else 0.0
        
        # 5. Logical progression
        # Check for increasing complexity in sentence structure
        sentence_lengths = [len(s.split()) for s in sentences]
        has_variation = max(sentence_lengths) - min(sentence_lengths) > 3
        progression_score = 0.1 if has_variation else 0.0
        
        # Calculate overall coherence score
        coherence_score = transition_score + tense_score + topic_score + paragraph_score + progression_score
        
        return min(1.0, coherence_score)
    
    def _evaluate_format_adherence(self, generation: str, instruction: str, complexity: str) -> float:
        """
        Evaluate adherence to requested format.
        
        Args:
            generation: Generated text
            instruction: Original instruction
            complexity: Complexity level of the instruction
            
        Returns:
            float: Format adherence score (0-1)
        """
        instruction_lower = instruction.lower()
        
        # Default score for non-format-specific instructions
        if complexity != "format_specific":
            return 0.7
        
        # Check for specific format requirements
        format_score = 0.5  # Start with moderate score
        
        # Table format
        if "table" in instruction_lower:
            # Check for table-like structure
            has_header_row = bool(re.search(r'^\s*\|[^|]+\|[^|]+\|', generation, re.MULTILINE))
            has_divider_row = bool(re.search(r'^\s*\|[\s\-]+\|[\s\-]+\|', generation, re.MULTILINE))
            has_data_rows = len(re.findall(r'^\s*\|', generation, re.MULTILINE)) > 2
            
            alternative_table = bool(re.search(r'^\s*[^\|:]+\s*:\s*[^\|:]+\s*$', generation, re.MULTILINE))
            
            if (has_header_row and has_data_rows) or alternative_table:
                format_score = 0.8
                format_score += 0.1 if has_divider_row else 0.0
            else:
                format_score = 0.2
        
        # Script/dialogue format
        elif "script" in instruction_lower or "dialogue" in instruction_lower:
            # Check for script format (CHARACTER: Dialogue)
            has_character_indicators = bool(re.search(r'^\s*[A-Z][a-zA-Z]+:', generation, re.MULTILINE))
            has_dialogue = '"' in generation or "'" in generation
            has_stage_directions = bool(re.search(r'\(.*?\)', generation))
            
            if has_character_indicators:
                format_score = 0.7
                format_score += 0.15 if has_dialogue else 0.0
                format_score += 0.15 if has_stage_directions else 0.0
            else:
                format_score = 0.3
        
        # List format
        elif "list" in instruction_lower:
            # Check for numbered or bulleted list
            has_numbered_items = bool(re.search(r'^\s*\d+\.', generation, re.MULTILINE))
            has_bulleted_items = bool(re.search(r'^\s*[•\-\*]', generation, re.MULTILINE))
            has_newline_items = generation.count('\n') >= 3
            
            if has_numbered_items or has_bulleted_items:
                format_score = 0.8
                format_score += 0.1 if has_newline_items else 0.0
            elif has_newline_items:
                # Has some list-like structure but no explicit numbering/bullets
                format_score = 0.5
            else:
                format_score = 0.2
        
        # Technical document format
        elif "technical document" in instruction_lower or "headings" in instruction_lower:
            # Check for headings and structure
            has_headings = bool(re.search(r'^\s*#+\s+.+$|^\s*[A-Z][A-Za-z\s]+:$', generation, re.MULTILINE))
            has_bullet_points = bool(re.search(r'^\s*[•\-\*]', generation, re.MULTILINE))
            has_code_examples = bool(re.search(r'`.*?`', generation))
            has_sections = generation.count('\n\n') >= 2
            
            format_score = 0.0
            format_score += 0.4 if has_headings else 0.0
            format_score += 0.2 if has_bullet_points else 0.0
            format_score += 0.2 if has_code_examples else 0.0
            format_score += 0.2 if has_sections else 0.0
        
        # Meeting agenda format
        elif "agenda" in instruction_lower or "time slots" in instruction_lower:
            # Check for time slots and structure
            has_time_slots = bool(re.search(r'\d+:\d+|^\s*\d+\s*(AM|PM|am|pm)', generation, re.MULTILINE))
            has_topics = bool(re.search(r'topic|discussion|agenda item', generation.lower()))
            has_structure = generation.count('\n') >= 5
            
            format_score = 0.0
            format_score += 0.4 if has_time_slots else 0.0
            format_score += 0.3 if has_topics else 0.0
            format_score += 0.3 if has_structure else 0.0
        
        return format_score
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on test results.
        
        Args:
            results: Test results by complexity level
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        # Check performance at each complexity level
        for complexity, complexity_results in results.items():
            score = complexity_results["average_following_score"]
            
            if score < 0.6:
                if complexity == "simple":
                    recommendations.append(
                        "Model struggles with even simple instructions. Consider basic fine-tuning "
                        "focused on instruction understanding and following."
                    )
                elif complexity == "medium":
                    recommendations.append(
                        "Model has difficulty with medium-complexity instructions. Focus on improving "
                        "ability to handle multi-part instructions with clearer structuring."
                    )
                elif complexity == "complex":
                    recommendations.append(
                        "Model struggles with complex instructions. Fine-tune with examples of "
                        "breaking down and systematically addressing multi-step instructions."
                    )
                elif complexity == "format_specific":
                    recommendations.append(
                        "Model fails to adhere to specified formats. Improve training on structured "
                        "content generation including tables, lists, and documents."
                    )
            elif score < 0.75:
                recommendations.append(
                    f"Model shows moderate performance on {complexity} instructions. "
                    f"Review specific failure cases to target improvements."
                )
        
        # Check specific components
        component_scores = {
            "completeness": [],
            "accuracy": [],
            "coherence": [],
            "format_adherence": []
        }
        
        for complexity_results in results.values():
            for test_result in complexity_results["test_results"]:
                for component, scores in component_scores.items():
                    if component in test_result:
                        scores.append(test_result[component])
        
        # Calculate average scores for each component
        component_averages = {}
        for component, scores in component_scores.items():
            if scores:
                component_averages[component] = np.mean(scores)
            else:
                component_averages[component] = 0.0
        
        # Add component-specific recommendations
        if component_averages.get("completeness", 0) < 0.7:
            recommendations.append(
                "Model often misses required elements in instructions. Improve training on "
                "identifying and addressing all parts of multi-part instructions."
            )
            
        if component_averages.get("accuracy", 0) < 0.7:
            recommendations.append(
                "Model responses are often inaccurate or inappropriate for the instruction type. "
                "Consider training specifically on recognizing instruction intent and generating "
                "appropriate response formats."
            )
            
        if component_averages.get("coherence", 0) < 0.7:
            recommendations.append(
                "Model generates text with poor coherence and logical flow. Improve training "
                "on structured writing with clear transitions and logical progression."
            )
            
        if component_averages.get("format_adherence", 0) < 0.7 and "format_specific" in results:
            recommendations.append(
                "Model has difficulty adhering to specified formats. Enhance training with "
                "examples of tables, scripts, lists, and structured documents."
            )
        
        # If no specific issues, add general recommendation
        if not recommendations:
            recommendations.append(
                "Model demonstrates good instruction following ability across different complexity levels. "
                "Consider testing with more domain-specific instructions to further validate performance."
            )
            
        return recommendations