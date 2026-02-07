"""MkDocs hooks to customize documentation build."""

import re


def on_post_page(output, page, config):
    """Post-process HTML to convert markdown bullet lists in tables to proper HTML lists.
    
    This runs after the page HTML is generated and converts any markdown-style
    bullet lists (- item) that appear in parameter tables to proper <ul>/<li> tags.
    """
    # Pattern to find parameter/attribute descriptions in tables that contain bullet lists
    # Look for content like "Description text:\n- item1\n- item2\n- item3"
    def fix_table_bullets(match):
        """Convert markdown bullets to HTML list in table cell."""
        content = match.group(1)
        
        # Check if this looks like it has bullet lists
        if '\n- ' not in content and not content.strip().startswith('-'):
            return match.group(0)  # No bullets, return unchanged
        
        lines = content.split('\n')
        new_lines = []
        in_list = False
        list_items = []
        preamble_lines = []
        
        for line in lines:
            # Check if line is a bullet
            bullet_match = re.match(r'^\s*- (.+)$', line)
            
            if bullet_match:
                # Bullet item
                if not in_list and preamble_lines:
                    # Start of list - emit preamble
                    new_lines.extend(preamble_lines)
                    preamble_lines = []
                    in_list = True
                
                list_items.append(bullet_match.group(1).strip())
                in_list = True
            else:
                # Not a bullet
                if in_list:
                    # End the list
                    new_lines.append('<ul>')
                    for item in list_items:
                        new_lines.append(f'<li>{item}</li>')
                    new_lines.append('</ul>')
                    list_items = []
                    in_list = False
                
                # Add this line
                if line.strip():
                    if in_list:
                        # Shouldn't happen, but handle it
                        preamble_lines.append(line)
                    else:
                        preamble_lines.append(line)
        
        # Handle remaining list items
        if in_list and list_items:
            if preamble_lines:
                new_lines.extend(preamble_lines)
                preamble_lines = []
            new_lines.append('<ul>')
            for item in list_items:
                new_lines.append(f'<li>{item}</li>')
            new_lines.append('</ul>')
        elif preamble_lines:
            new_lines.extend(preamble_lines)
        
        result = '\n'.join(new_lines)
        return f'<p>{result}</p>'
    
    # Find all <p> tags inside table cells that might contain bullet lists
    output = re.sub(
        r'<p>([^<]*?(?:\n- [^\n]+)+[^<]*?)</p>',
        fix_table_bullets,
        output,
        flags=re.MULTILINE
    )
    
    return output
