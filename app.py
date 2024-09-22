from flask import Flask, request, redirect, render_template
import sqlite3
import string
import random

app = Flask(__name__)

# Function to generate a random string for the short URL
def generate_short_url(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Database setup
def init_db():
    conn = sqlite3.connect('url_shortener.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS urls
                 (id INTEGER PRIMARY KEY, original_url TEXT, short_url TEXT)''')
    conn.commit()
    conn.close()

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        original_url = request.form['url']
        short_url = generate_short_url()
        
        conn = sqlite3.connect('url_shortener.db')
        c = conn.cursor()
        c.execute("INSERT INTO urls (original_url, short_url) VALUES (?, ?)", (original_url, short_url))
        conn.commit()
        conn.close()
        
        return render_template('index.html', short_url=short_url)
    
    return render_template('index.html')

# Route to redirect to the original URL
@app.route('/<short_url>')
def redirect_to_url(short_url):
    conn = sqlite3.connect('url_shortener.db')
    c = conn.cursor()
    c.execute("SELECT original_url FROM urls WHERE short_url=?", (short_url,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return redirect(row[0])
    return "URL not found!", 404

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
