# Import required libraries
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt


def read_excel_to_df(excel_file):
    try:
        # Read Excel file directly into DataFrame
        df = pd.read_excel(excel_file)
        # Display the DataFrame
        print("DataFrame contents:")        
        return df
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def add_month_column(df):
    # Split ANO_MES column into year and month
    df[['Year', 'Month']] = df['ANO_MES'].str.split('_', expand=True)
    return df

def get_total_cases(df):
    # Calculate total cases for each category
    total_cases = {
        'Category': ['Hombre', 'Indeterminado', 'Mujer'],
        'Total': [
            df['Hombre'].sum(),
            df['Indeterminado'].sum(),
            df['Mujer'].sum()
        ]
    }
    return pd.DataFrame(total_cases)

def get_yearly_cases(df):
    # Extract year from ANO_MES
    df['Year'] = df['ANO_MES'].str.split('_').str[0]
    
    # Group by year and sum cases
    yearly_cases = df.groupby('Year').agg({
        'Hombre': 'sum',
        'Indeterminado': 'sum',
        'Mujer': 'sum'
    }).reset_index()
    
    return yearly_cases

def plot_total_cases(total_df):
    plt.figure(figsize=(10, 6))
    plt.bar(total_df['Category'], total_df['Total'])
    plt.title('Total HIV Cases by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Guardar el gráfico
    plt.savefig('total_cases.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_yearly_cases(yearly_df):
    plt.figure(figsize=(12, 6))
    
    x = range(len(yearly_df['Year']))
    width = 0.25
    
    plt.bar(x, yearly_df['Hombre'], width, label='Hombre')
    plt.bar([i + width for i in x], yearly_df['Indeterminado'], width, label='Indeterminado')
    plt.bar([i + width*2 for i in x], yearly_df['Mujer'], width, label='Mujer')
    
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    plt.title('Yearly HIV Cases by Category')
    plt.xticks([i + width for i in x], yearly_df['Year'], rotation=45)
    plt.legend()
    plt.tight_layout()
     # Guardar el gráfico
    plt.savefig('yearly_cases.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
	# Example usage
	excel_file = "Notificaciones_VIH_2010_2019.xlsx"  # Replace with your Excel file path
	# Read the dataframe (assuming it's already loaded as df)
	# Call the function
	df = read_excel_to_df(excel_file)
	print(df)
	# Process dataframes
	df_with_month = add_month_column(df)
	total_cases_df = get_total_cases(df)
	yearly_cases_df = get_yearly_cases(df)
    
    # Print results
	print("\nDataframe with Month column:")
	print(df_with_month.head())
	
	print("\nTotal cases by category:")
	print(total_cases_df)
	
	print("\nYearly cases by category:")
	print(yearly_cases_df)
	
	# Plot results
	plot_total_cases(total_cases_df)
	plot_yearly_cases(yearly_cases_df)

	
	

