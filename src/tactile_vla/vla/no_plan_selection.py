"""Select source-supervised samples without counterfactual variants."""


def select_factual(rows, stream):
    selected = []
    for index, global_index in zip(stream['manifest_row_indices'], stream['global_indices'], strict=True):
        index, global_index = int(index), int(global_index)
        row = rows[index]
        if int(row['global_index']) != global_index:
            raise ValueError('Manifest row and selected global index disagree')
        if (row.get('sample_variant', 'factual') == 'factual'
                and row.get('need_variant', 'real') == 'real'
                and not row.get('need_counterfactual')):
            selected.append((index, global_index))
    if not selected:
        raise ValueError('No factual samples remain')
    return [pair[0] for pair in selected], [pair[1] for pair in selected]
