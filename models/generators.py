from pathlib import Path

def generate_loop_mealy_dot(size: int, unknowns: int, output_dir: Path = None) -> str:
    """
    Generate a DOT representation of a loop Mealy machine.
    """
    if unknowns > size:
        raise ValueError(f"unknowns ({unknowns}) cannot be greater than size ({size})")
    
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    
    dot_lines = ["digraph g {", '__start0 [label="" shape="none"];', ""]
    
    for i in range(size):
        dot_lines.append(f'    s{i} [shape="circle" label="{i}"];')
    
    for i in range(size):
        next_state = (i + 1) % size
        output = "unknown" if i >= size - unknowns else str(i + 1)
        dot_lines.append(f'    s{i} -> s{next_state} [label="i/{output}"];')
    
    dot_lines.extend(["", "__start0 -> s0;", "}"])
    dot_string = "\n".join(dot_lines)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"loop_s{size}_u{unknowns}.dot"
        with open(filepath, 'w') as f:
            f.write(dot_string)
        print(f"Generated: {filepath}")
    
    return dot_string

def generate_multiple_loop_machines(size_configs: list, output_dir: Path = None):
    """Generate multiple loop machines."""
    for size, unknowns in size_configs:
        generate_loop_mealy_dot(size, unknowns, output_dir)

def generate_adder_mealy_dot(alphabet_max: int = 10, output_dir: Path = None) -> str:
    """
    Generate a DOT representation of a 2-step adder Mealy machine.
    """
    if alphabet_max < 2:
        raise ValueError(f"alphabet_max must be at least 2, got {alphabet_max}")
    
    dot_lines = ["digraph g {", '__start0 [label="" shape="none"];', ""]
    
    for i in range(alphabet_max + 1):
        dot_lines.append(f'    s{i} [shape="circle" label="{i}"];')
    
    for input_val in range(1, alphabet_max + 1):
        dot_lines.append(f'    s0 -> s{input_val} [label="{input_val}/0"];')
    
    for first_num in range(1, alphabet_max + 1):
        for second_num in range(1, alphabet_max + 1):
            total = first_num + second_num
            output = str(total) if total <= alphabet_max else "unknown"
            dot_lines.append(f'    s{first_num} -> s0 [label="{second_num}/{output}"];')
    
    dot_lines.extend(["", "__start0 -> s0;", "}"])
    dot_string = "\n".join(dot_lines)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"adder_max{alphabet_max}.dot"
        with open(filepath, 'w') as f:
            f.write(dot_string)
        print(f"Generated: {filepath}")
    
    return dot_string

def generate_boolean_stack_dot(max_depth: int, output_dir: Path = None) -> str:
    """Generate a DOT representation of a bounded boolean stack Mealy machine."""
    
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, got {max_depth}")
    
    total_states = sum(2**d for d in range(max_depth + 1))

    state_to_depth = {}
    depth_to_states = {}
    
    state_counter = 0
    for depth in range(max_depth + 1):
        depth_to_states[depth] = []
        num_states_at_depth = 2 ** depth
        for i in range(num_states_at_depth):
            state_to_depth[state_counter] = (depth, i)
            depth_to_states[depth].append(state_counter)
            state_counter += 1
    
    dot_lines = []
    dot_lines.append("digraph g {")
    dot_lines.append('__start0 [label="" shape="none"];')
    dot_lines.append("")
    
    for state_id in range(total_states):
        depth, index = state_to_depth[state_id]
        dot_lines.append(f'    s{state_id} [shape="circle" label="{state_id}"];')
    
    dot_lines.append("")
    
    for state_id in range(total_states):
        depth, index = state_to_depth[state_id]
        
        if depth == 0:
            pop_output = "x"
            next_state_pop = state_id
        elif depth == max_depth:
            pop_output = "unknown"
            parent_index = index // 2
            parent_state = depth_to_states[depth - 1][parent_index]
            next_state_pop = parent_state
        else:
            pop_output = str(index % 2)
            parent_index = index // 2
            parent_state = depth_to_states[depth - 1][parent_index]
            next_state_pop = parent_state
        
        dot_lines.append(f'    s{state_id} -> s{next_state_pop} [label="pop/{pop_output}"];')
        
        if depth < max_depth:
            child_index = index * 2 + 0
            next_state_push0 = depth_to_states[depth + 1][child_index]
            dot_lines.append(f'    s{state_id} -> s{next_state_push0} [label="push0/v"];')
        else:
            dot_lines.append(f'    s{state_id} -> s{state_id} [label="push0/v"];')
        
        if depth < max_depth:
            child_index = index * 2 + 1
            next_state_push1 = depth_to_states[depth + 1][child_index]
            dot_lines.append(f'    s{state_id} -> s{next_state_push1} [label="push1/v"];')
        else:
            dot_lines.append(f'    s{state_id} -> s{state_id} [label="push1/v"];')
    
    dot_lines.append("")
    dot_lines.append("__start0 -> s0;")
    dot_lines.append("}")
    
    dot_string = "\n".join(dot_lines)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"boolean_stack_d{max_depth}.dot"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(dot_string)
        
        print(f"Generated: {filepath}")
    
    return dot_string