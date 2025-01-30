import os


def delete_bank_plots():

    sufolders = ['banks', 'ecl', 'exposure', 'lgd', 'macro', 'tms', 'zscores']

    for subfolder in sufolders:
        # loop through '../../plots/subfolder/' and delete all *.png files in all subfolders
        for root, dirs, files in os.walk(f'../../plots/{subfolder}/'):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))

delete_bank_plots()
