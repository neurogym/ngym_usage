"""Programmatically generate jupyte notebooks."""

from pathlib import Path
import inspect
import importlib

import nbformat as nbf

import neurogym as ngym


def get_linenumber(m):
    """Get line number of a member."""
    try:
        return inspect.findsource(m[1])[1]
    except AttributeError:
        return -1

# Get functions from module
modulename = 'train_and_analysis_template'
module = importlib.import_module(modulename)
members = inspect.getmembers(module)  # get all members
# members = [m for m in members if inspect.isfunction(m[1])]  # keep all functions
m_tmp = list()
for m in members:
    try:
        tmp_module = m[1].__module__
        if tmp_module == modulename:
            m_tmp.append(m)
    except AttributeError:
        pass

members = m_tmp          
members.sort(key=get_linenumber)  # sort by line number


def func_to_script(code):
    # Remove indentation
    code = code.replace('\n    ', '\n')
    # Remove first line
    ind = code.find('\n')
    assert code[ind-1] == ':'  # end of def should be this
    code = code[ind+1:]
    
    # Search if there is a return
    ind = code.find('return')
    # Check this is the only return
    assert code.find('return', ind + 1) == -1
    
    # Remove the return line
    code = code[:ind]
    
    return code


def auto_generate_notebook(envid):
    nb = nbf.v4.new_notebook()
    # Initial code block
    with open(modulename + '.py', 'r') as f:
        codelines = f.readlines()
    # Everything before first function/class
    code = ''.join(codelines[:get_linenumber(members[0])])
    code = code + "envid = '{:s}'".format(envid)
    cells = [nbf.v4.new_code_cell(code)]

    func_to_script_list = ['train_network', 'run_network']
    for name, obj in members:
        code = inspect.getsource(obj)

        if name in func_to_script_list:
            # Turn function into script
            code = func_to_script(code)

        if name.find('analysis_') == 0:  # starts with this
            # Add a line that's running this function
            code = code + '\n' + code[4:code.find('\n') - 1] # 4 for "def "

        cells.append(nbf.v4.new_code_cell(code))

    nb['cells'] = cells

    #     nb['cells'] = [nbf.v4.new_markdown_cell(text),
    #                nbf.v4.new_code_cell(code) ]

    fname = Path('.') / 'auto_notebooks' / (envid + '.ipynb')
    nbf.write(nb, fname)


if __name__ == '__main__':
    all_envs = ngym.all_envs(tag='supervised')
    for envid in all_envs:
        auto_generate_notebook(envid)
