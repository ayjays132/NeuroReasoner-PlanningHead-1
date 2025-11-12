"""
Final comprehensive test with excellent prompts to showcase model capabilities
"""

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import json
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 100)
print("FINAL COMPREHENSIVE TEST - Showcasing Best Outputs")
print("=" * 100)

# Load model from local directory
print("\n[LOADING] Loading model from local directory...")
model = AutoModel.from_pretrained(
    ".",
    trust_remote_code=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(".", local_files_only=True)
model.set_tokenizer(tokenizer)
model.eval()
print("[OK] Model loaded successfully!\n")

# Excellent test prompts
test_cases = [
    {
        "name": "Complex Strategic Planning",
        "prompt": """<context>
A technology startup needs to scale from 10 to 100 employees over 18 months while maintaining culture, 
hiring top talent, securing Series B funding, and launching 3 new products.
</context>

<plan>
1. Establish hiring framework with clear role definitions and cultural fit criteria
2. Build scalable infrastructure (HR systems, onboarding, performance management)
3. Develop product roadmap with clear milestones and resource allocation
4. Create investor pitch materials and financial projections
5. Implement culture preservation mechanisms (values documentation, regular all-hands)
6. Set up mentorship and career development programs
7. Establish metrics and KPIs for tracking growth and culture health
</plan>

<reasoning>
This requires balancing rapid growth with maintaining quality and culture. Each component must be 
coordinated - hiring affects product development, funding enables hiring, culture affects retention.
</reasoning>

The strategic execution plan involves:"""
    },
    {
        "name": "Deep Philosophical Reasoning",
        "prompt": """<analyze>
What is the relationship between consciousness, intelligence, and planning?
</analyze>

<reasoning>
Consciousness involves subjective experience and self-awareness. Intelligence involves problem-solving 
and adaptation. Planning involves projecting future states and coordinating actions.
</reasoning>

<contemplate>
If consciousness requires self-awareness, and planning requires projecting oneself into future 
scenarios, then planning might be a form of temporal self-awareness - awareness of oneself across time.
</contemplate>

<synthesize>
The relationship between these concepts suggests:"""
    },
    {
        "name": "Creative Problem-Solving",
        "prompt": """<plan>
1. Question all assumptions about the problem
2. Explore analogies from completely different domains
3. Consider what would happen if constraints were removed
4. Combine seemingly incompatible approaches
5. Think from first principles
</plan>

The problem: How can we make education more effective while reducing costs and increasing accessibility?

<internal_thinking>
Most solutions assume traditional classroom models. What if we completely reimagined the structure 
of learning itself? What if we looked at how people actually learn best, regardless of traditional 
educational frameworks?
</internal_thinking>

An innovative solution that breaks conventional assumptions might:"""
    },
    {
        "name": "Meta-Cognitive Reflection",
        "prompt": """<reflect>
How should I approach planning when the future is uncertain and requirements are constantly changing?
</reflect>

<internal_thinking>
Traditional planning assumes predictability. But in uncertain environments, the planning process 
itself must be adaptive. I need to plan for planning - create a meta-planning framework.
</internal_thinking>

<contemplate>
The key insight is that in uncertain environments, planning becomes less about predicting the future 
and more about creating flexible structures that can adapt.
</contemplate>

The meta-planning approach involves:"""
    }
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*100}")
    print(f"TEST {i}: {test['name']}")
    print(f"{'='*100}")
    print(f"\nPROMPT:\n{test['prompt'][:200]}...\n")
    
    inputs = tokenizer(test['prompt'], return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        
        # Check for plan vector
        plan_info = None
        if outputs.get('plan_vector') is not None:
            plan_vec = outputs['plan_vector'][0]
            plan_info = {
                "extracted": True,
                "shape": list(plan_vec.shape),
                "norm": float(torch.norm(plan_vec).item()),
                "stats": {
                    "min": float(plan_vec.min().item()),
                    "max": float(plan_vec.max().item()),
                    "mean": float(plan_vec.mean().item())
                }
            }
        
        # Load generation config and use it
        try:
            gen_config = GenerationConfig.from_pretrained(".", local_files_only=True)
            # Override with specific values for this generation
            gen_config.max_length = inputs['input_ids'].shape[1] + 300
            gen_config.min_length = inputs['input_ids'].shape[1] + 50
            gen_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            gen_config.eos_token_id = tokenizer.eos_token_id
        except:
            # Fallback if config can't be loaded
            gen_config = None
        
        # Generate using config or fallback parameters
        if gen_config:
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=gen_config,
            )
        else:
            # Fallback with basic parameters
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + 300,
                min_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        if hasattr(generated, 'sequences'):
            generated_ids = generated.sequences[0].cpu().tolist()
        else:
            generated_ids = generated[0].cpu().tolist()
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_only = tokenizer.decode(generated_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        
        # Clean output for display (handle Unicode)
        output_display = generated_only[:500].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(f"\nOUTPUT:\n{output_display}...")
        
        results.append({
            "name": test['name'],
            "prompt": test['prompt'],
            "full_output": generated_text,
            "generated_only": generated_only,
            "plan_vector": plan_info
        })

# Save results
with open('final_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*100}")
print("TEST COMPLETE - Results saved to final_test_results.json")
print("=" * 100)

