"""
Compare two expressions to see if they're equivalent
"""

from collections import defaultdict

def parse_expression_1(expr_str):
    """Parse the format from uwg_intersect_uwg_e.py"""
    # Format: "e_1 e_2 e_3 e_4 + -1 e_3 e_3 e_4 + ..."
    terms = defaultdict(int)

    parts = expr_str.split(' + ')
    for part in parts:
        tokens = part.split()

        # Extract coefficient
        if tokens[0].lstrip('-').isdigit():
            coeff = int(tokens[0])
            indices = []
            for token in tokens[1:]:
                if token.startswith('e_'):
                    indices.append(int(token[2:]))
        else:
            coeff = 1
            indices = []
            for token in tokens:
                if token.startswith('e_'):
                    indices.append(int(token[2:]))

        key = tuple(sorted(indices))
        terms[key] += coeff

    return terms

def parse_expression_2(expr_str):
    """Parse the SymPy format from e_algebra.py"""
    # Format: "e_1*e_2*e_3*e_4 - e_1*e_2*e_7 - ..."
    import re
    terms = defaultdict(int)

    # Replace - with + - for easier splitting
    expr_str = expr_str.replace(' - ', ' + -')
    parts = expr_str.split(' + ')

    for part in parts:
        part = part.strip()

        coeff = 1
        indices = []

        # Check if the whole term starts with a minus sign
        if part.startswith('-'):
            # Check if it's -e_... (implicit -1 coefficient) or -<number>*e_...
            if re.match(r'^-e_', part):
                coeff = -1
                part = part[1:]  # Remove the leading minus
            elif re.match(r'^-\d+\*', part):
                # Extract coefficient like -2*
                coeff_match = re.match(r'^(-?\d+)\*', part)
                coeff = int(coeff_match.group(1))
                part = part[coeff_match.end():]
        else:
            # First extract any leading coefficient
            # Match patterns like "2*" or "6*" at the start
            coeff_match = re.match(r'^(\d+)\*', part)
            if coeff_match:
                coeff = int(coeff_match.group(1))
                part = part[coeff_match.end():]

        # Now find all e_i or e_i**n patterns
        # Pattern: e_<number> optionally followed by **<number>
        e_pattern = r'e_(\d+)(?:\*\*(\d+))?'
        matches = re.findall(e_pattern, part)

        for idx_str, exp_str in matches:
            idx = int(idx_str)
            exp = int(exp_str) if exp_str else 1
            indices.extend([idx] * exp)

        key = tuple(sorted(indices))
        terms[key] += coeff

    return terms

# Test expressions
expr1 = "e_1 e_2 e_3 e_4 + -1 e_3 e_3 e_4 + -2 e_2 e_4 e_4 + -3 e_2 e_3 e_5 + -1 e_1 e_4 e_5 + 3 e_5 e_5 + 8 e_4 e_6 + 6 e_1 e_9 + -48 e_10 + -2 e_1 e_3 e_6 + 11 e_3 e_7 + -1 e_1 e_2 e_7 + 6 e_2 e_8"

expr2 = "e_1*e_2*e_3*e_4 - e_1*e_2*e_7 - 2*e_1*e_3*e_6 - e_1*e_4*e_5 + 6*e_1*e_9 - 48*e_10 - 3*e_2*e_3*e_5 - 2*e_2*e_4**2 + 6*e_2*e_8 + 11*e_3*e_7 - e_3**2*e_4 + 8*e_4*e_6 + 3*e_5**2"

terms1 = parse_expression_1(expr1)
terms2 = parse_expression_2(expr2)

print("Expression 1 terms:")
for key in sorted(terms1.keys()):
    print(f"  {terms1[key]} * e_{{{', '.join(map(str, key))}}}")

print("\nExpression 2 terms:")
for key in sorted(terms2.keys()):
    print(f"  {terms2[key]} * e_{{{', '.join(map(str, key))}}}")

print("\n" + "="*60)
if terms1 == terms2:
    print("YES! The expressions are IDENTICAL.")
else:
    print("NO! The expressions are DIFFERENT.")

    # Show differences
    all_keys = set(terms1.keys()) | set(terms2.keys())
    for key in sorted(all_keys):
        c1 = terms1.get(key, 0)
        c2 = terms2.get(key, 0)
        if c1 != c2:
            print(f"  Difference at e_{{{', '.join(map(str, key))}}}: {c1} vs {c2}")
