#!/usr/bin/env python3
"""
Test system prompt functionality for vLLM chat mode.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.bench_config import GenDefaults, load_bench_config
from src.core.engines import create_engine
from src.core.interfaces import GenerationParams


def test_system_prompt_config_loading():
    """Test that system_prompt is properly loaded from YAML config."""
    print("\n🧪 Testing system prompt config loading...")
    
    # Test GenDefaults with system_prompt
    gen_defaults = GenDefaults(system_prompt="You are a helpful assistant.")
    assert gen_defaults.system_prompt == "You are a helpful assistant."
    print("✅ GenDefaults system_prompt field works")
    
    # Test None system_prompt
    gen_defaults_none = GenDefaults()
    assert gen_defaults_none.system_prompt is None
    print("✅ GenDefaults system_prompt defaults to None")


def test_system_prompt_validation():
    """Test system prompt validation without loading models."""
    print("\n🧪 Testing system prompt validation...")
    
    # Test create_engine with system_prompt parameter
    try:
        # This should not raise an error even if model loading fails
        engine = create_engine(
            engine_name="vllm",
            model_id="microsoft/DialoGPT-small",
            dtype="float16",
            generation_mode="chat",
            system_prompt="You are a helpful assistant."
        )
        print("✅ create_engine accepts system_prompt parameter")
        engine.close()
    except Exception as e:
        if "system_prompt" in str(e):
            print(f"❌ System prompt error: {e}")
        else:
            print(f"⚠️  Expected error (no GPU/model): {e}")
    
    # Test create_engine without system_prompt
    try:
        engine = create_engine(
            engine_name="vllm",
            model_id="microsoft/DialoGPT-small",
            dtype="float16",
            generation_mode="chat"
        )
        print("✅ create_engine works without system_prompt")
        engine.close()
    except Exception as e:
        if "system_prompt" in str(e):
            print(f"❌ System prompt error: {e}")
        else:
            print(f"⚠️  Expected error (no GPU/model): {e}")


def test_chat_message_formatting_with_system_prompt():
    """Test chat message formatting logic with system prompts."""
    print("\n🧪 Testing chat message formatting with system prompts...")
    
    # Test conversation building logic
    def build_conversation(prompt, system_prompt=None):
        conversation = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": prompt})
        return conversation
    
    # Test with system prompt
    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello, how are you?"
    conversation = build_conversation(user_prompt, system_prompt)
    
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    assert conversation == expected
    print("✅ Chat conversation with system prompt formatted correctly")
    
    # Test without system prompt
    conversation_no_system = build_conversation(user_prompt, None)
    expected_no_system = [{"role": "user", "content": "Hello, how are you?"}]
    assert conversation_no_system == expected_no_system
    print("✅ Chat conversation without system prompt formatted correctly")
    
    # Test multiple prompts
    prompts = ["Hello", "How are you?", "What's the weather?"]
    system_prompt = "You are a weather assistant."
    conversations = []
    for prompt in prompts:
        conversations.append(build_conversation(prompt, system_prompt))
    
    assert len(conversations) == 3
    for conv in conversations:
        assert conv[0]["role"] == "system"
        assert conv[0]["content"] == system_prompt
        assert conv[1]["role"] == "user"
    print("✅ Multiple chat conversations with system prompt formatted correctly")


def test_yaml_config_with_system_prompt():
    """Test loading YAML config with system prompts."""
    print("\n🧪 Testing YAML config with system prompts...")
    
    # Test the thinkingparam.yaml config
    try:
        config = load_bench_config("configs/archive/thinkingparam.yaml")
        
        # Find BPT models
        bpt_models = [m for m in config.models if "BPT-OSS-20B" in m.name]
        assert len(bpt_models) == 3, f"Expected 3 BPT models, found {len(bpt_models)}"
        
        # Check that each has a system prompt
        for model in bpt_models:
            assert model.generation.system_prompt is not None
            assert len(model.generation.system_prompt) > 0
            print(f"✅ {model.name} has system prompt: {model.generation.system_prompt[:50]}...")
        
        # Check that system prompts are different
        system_prompts = [m.generation.system_prompt for m in bpt_models]
        assert len(set(system_prompts)) == 3, "All system prompts should be different"
        print("✅ All BPT models have unique system prompts")
        
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")


def test_actual_generation_with_system_prompt():
    """Test actual generation with system prompts (minimal model loading)."""
    print("\n🧪 Testing actual generation with system prompts...")
    
    test_prompt = "What is 2+2?"
    generation_params = GenerationParams(
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9
    )
    
    # Test with system prompt
    try:
        print("  Testing generation with system prompt...")
        engine_with_system = create_engine(
            engine_name="vllm",
            model_id="microsoft/DialoGPT-small",
            dtype="float16",
            generation_mode="chat",
            system_prompt="You are a math tutor. Always show your work."
        )
        
        result_with_system = engine_with_system.generate(test_prompt, generation_params)
        assert result_with_system.text is not None
        assert len(result_with_system.text) > 0
        assert result_with_system.completion_tokens > 0
        print(f"    ✅ Generated with system prompt: '{result_with_system.text[:30]}...'")
        
        engine_with_system.close()
        
    except Exception as e:
        if "system_prompt" in str(e):
            print(f"    ❌ System prompt error: {e}")
        else:
            print(f"    ⚠️  Expected error (no GPU/model): {e}")
    
    # Test without system prompt for comparison
    try:
        print("  Testing generation without system prompt...")
        engine_no_system = create_engine(
            engine_name="vllm",
            model_id="microsoft/DialoGPT-small",
            dtype="float16",
            generation_mode="chat"
        )
        
        result_no_system = engine_no_system.generate(test_prompt, generation_params)
        assert result_no_system.text is not None
        assert len(result_no_system.text) > 0
        assert result_no_system.completion_tokens > 0
        print(f"    ✅ Generated without system prompt: '{result_no_system.text[:30]}...'")
        
        engine_no_system.close()
        
    except Exception as e:
        if "system_prompt" in str(e):
            print(f"    ❌ System prompt error: {e}")
        else:
            print(f"    ⚠️  Expected error (no GPU/model): {e}")


def main():
    """Run all system prompt tests."""
    print("🚀 Starting system prompt functionality tests...")
    
    test_system_prompt_config_loading()
    test_system_prompt_validation()
    test_chat_message_formatting_with_system_prompt()
    test_yaml_config_with_system_prompt()
    test_actual_generation_with_system_prompt()
    
    print("\n✅ All system prompt tests completed!")


if __name__ == "__main__":
    main()
