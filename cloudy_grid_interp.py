from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module


def toChemistryIon(AstroIon):
    element, roman_numeral = split_string_by_second_uppercase(AstroIon)
    number = fromRoman(roman_numeral)
    return element + str('+') + str(number - 1)

def toAstroIon(ChemistryIon):
    split = ChemistryIon.split('+')

    if split[1] == '':
        ion = 2
    else:
        ion = int(split[1]) + 1

    return split[0] + toRoman(ion)

def gen_input_text(redshift, metal, hden, stop_neutral):
    file_name_prefix = 'z=' + str(redshift) + 'Z=' + str(metal) + 'nh=' + str(hden) + 'NHI=' + str(stop_neutral)
    input_text = """title CGM
cmb redshift """ + str(redshift) + """
# table HM12 z=0.12
table KS19 redshift 0.1502
metals """ + str(metal) + """ log #log Z/Zsun
hden """ + str(hden) + """ log #log nH * cm^3
radius 30
set temperature floor 3.7
stop neutral column """ + str(stop_neutral) + """ log # log HI * cm^2
double optical depths
set trim -20
save species column density last \"""" + str(file_name_prefix) + ".col" + """\" all
iterate to convergence
"""
    
    return input_text, file_name_prefix


def writeColumnsFile(output_dir, redshift_range, metal_range, hden_range, neutral_column_range, extra=None, ion_names=None):
    if ion_names is not None:
        ions_tuple = ('z', 'Z', 'n_H') + tuple(ion_names)
        full = QTable(names=ions_tuple)
    else:
        full = QTable(names=('z', 'Z', 'n_H', 'HI', 'CII', 'CIII','CIV', 'OI', 'OII', 'OIII', 'OVI', 'NIII', 'NV', 'SiII', 'SiIII', 'SiIV'))

    for l, redshift in enumerate(redshift_range):
        print('redshift = ', redshift)
        for i, metal in enumerate(metal_range):
            print('metal = ', metal)
            for j, hden in enumerate(hden_range):
                for k, stop_neutral in enumerate(neutral_column_range):
                    input_text, file_name_prefix = gen_input_text(redshift, metal, hden, stop_neutral)
                    file_name = output_dir + file_name_prefix + '.col'

                    t = Table.read(file_name, format='ascii.commented_header', delimiter='	')

                    N_HI = t['column density H'][0]
                    N_CIV = t['C+3'][0]
                    N_CIII = t['C+2'][0]
                    N_CII = t['C+'][0]
                    N_OI = t['O'][0]
                    N_OII = t['O+'][0]
                    N_OIII = t['O+2'][0]
                    N_OVI = t['O+5'][0]
                    N_NIII = t['N+2'][0]
                    N_NV = t['N+4'][0]
                    N_SiII = t['Si+'][0]
                    N_SiIII = t['Si+2'][0]
                    N_SiIV = t['Si+3'][0]

                    full.add_row([redshift, metal, hden, N_HI, N_CII, N_CIII, N_CIV, N_OI, N_OII, N_OIII, N_OVI, N_NIII, N_NV, N_SiII, N_SiIII, N_SiIV])

    redshift_min = np.round(min(redshift_range), decimals=2)
    redshift_max = np.round(max(redshift_range), decimals=2)
    metal_min = np.round(min(metal_range), decimals=2)
    metal_max = np.round(max(metal_range), decimals=2)
    hden_min = np.round(min(hden_range), decimals=2)
    hden_max = np.round(max(hden_range), decimals=2)
    neutral_min = np.round(min(neutral_column_range), decimals=2)
    neutral_max = np.round(max(neutral_column_range), decimals=2)
    if extra is None:
        file_name = output_dir + 'full__' + \
            'z=' + str(redshift_min) + '_' + str(redshift_max) + '__' + \
            'Z=' + str(metal_min) + '_' + str(metal_max) + '__' + \
            'nh=' + str(hden_min) + str(hden_max) + '__' + \
            'NHI=' + str(neutral_min) + '_' + str(neutral_max) + '.dat'
    else:
        file_name = output_dir + 'full__' + \
            'z=' + str(redshift_min) + '_' + str(redshift_max) + '__' + \
            'Z=' + str(metal_min) + '_' + str(metal_max) + '__' + \
            'nh=' + str(hden_min) + str(hden_max) + '__' + \
            'NHI=' + str(neutral_min) + '_' + str(neutral_max) + str('__') + str(extra) + '.dat'
        
    ascii.write(full, file_name, overwrite=True)



def readColumnsFile(output_dir, redshift_range, metal_range, hden_range, neutral_column_range, extra=None):
    redshift_min = np.round(min(redshift_range), decimals=2)
    redshift_max = np.round(max(redshift_range), decimals=2)
    metal_min = np.round(min(metal_range), decimals=2)
    metal_max = np.round(max(metal_range), decimals=2)
    hden_min = np.round(min(hden_range), decimals=2)
    hden_max = np.round(max(hden_range), decimals=2)
    neutral_min = np.round(min(neutral_column_range), decimals=2)
    neutral_max = np.round(max(neutral_column_range), decimals=2)

    if extra is None:
        file_name = output_dir + 'full__' + \
            'z=' + str(redshift_min) + '_' + str(redshift_max) + '__' + \
            'Z=' + str(metal_min) + '_' + str(metal_max) + '__' + \
            'nh=' + str(hden_min) + str(hden_max) + '__' + \
            'NHI=' + str(neutral_min) + '_' + str(neutral_max) + '.dat'
    else:
        file_name = output_dir + 'full__' + \
            'z=' + str(redshift_min) + '_' + str(redshift_max) + '__' + \
            'Z=' + str(metal_min) + '_' + str(metal_max) + '__' + \
            'nh=' + str(hden_min) + str(hden_max) + '__' + \
            'NHI=' + str(neutral_min) + '_' + str(neutral_max) + str('__') + str(extra) + '.dat'
        
    full = ascii.read(file_name)

    return full

def genInterpolatedGridFunction(output_dir, redshift_range, metal_range, hden_range, neutral_column_range, extra=None):
    full = readColumnsFile(output_dir, redshift_range, metal_range, hden_range, neutral_column_range, extra=extra)

    empty = []
    shape = (len(redshift_range), len(metal_range), len(hden_range), len(neutral_column_range))

    ions = []
    # Iterate through columns
    for column in full.itercols():

        if (('z' in column.name) or ('Z' in column.name)) or ('n_H' in column.name):
            empty.append(np.array(column).reshape(shape))
        else:
            ions.append(column.name)
            empty.append(np.log10(np.array(column)).reshape(shape))

    values = np.stack(empty, axis=4)
    points = (redshift_range, metal_range, hden_range, neutral_column_range)

    output = RegularGridInterpolator(points, values)

    return output