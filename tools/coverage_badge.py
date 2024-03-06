from bs4 import BeautifulSoup

html_file = "coverage/index.html"

colour_dict = {'coverage-low':"red", 'coverage-medium':"yellow", 'coverage-high':"brightgreen"}

with open(html_file, 'r') as file:
    soup = BeautifulSoup(file, 'html.parser')

    # Find the "coverage" table
    coverage_table = soup.find('table', {'class': 'coverage'})

    # Find the second row of the "coverage" table
    second_row = coverage_table.find_all('tr')[1]

    # Find the third column of the second row
    third_column = second_row.find_all('td')[2]

    percentage = third_column.get_text()
    
    # Get the class of the <td> element
    td_class = third_column.get('class')[0]

print(int(float(percentage.replace("%", ""))))
