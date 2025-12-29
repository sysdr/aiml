#!/bin/bash

echo "🔧 Setting up Day 51: Spam Detection Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Download spam dataset
echo "📊 Downloading spam dataset..."
if [ ! -f "spambase.data" ]; then
    wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
    echo "✅ Dataset downloaded: spambase.data"
else
    echo "✅ Dataset already exists: spambase.data"
fi

# Create column names file
cat > spambase.names << 'NAMES'
word_freq_make,word_freq_address,word_freq_all,word_freq_3d,word_freq_our,word_freq_over,word_freq_remove,word_freq_internet,word_freq_order,word_freq_mail,word_freq_receive,word_freq_will,word_freq_people,word_freq_report,word_freq_addresses,word_freq_free,word_freq_business,word_freq_email,word_freq_you,word_freq_credit,word_freq_your,word_freq_font,word_freq_000,word_freq_money,word_freq_hp,word_freq_hpl,word_freq_george,word_freq_650,word_freq_lab,word_freq_labs,word_freq_telnet,word_freq_857,word_freq_data,word_freq_415,word_freq_85,word_freq_technology,word_freq_1999,word_freq_parts,word_freq_pm,word_freq_direct,word_freq_cs,word_freq_meeting,word_freq_original,word_freq_project,word_freq_re,word_freq_edu,word_freq_table,word_freq_conference,char_freq_semicolon,char_freq_parenthesis,char_freq_bracket,char_freq_exclamation,char_freq_dollar,char_freq_hash,capital_run_length_average,capital_run_length_longest,capital_run_length_total,is_spam
NAMES

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Run: python lesson_code.py"
echo "   2. Run tests: pytest test_lesson.py -v"
echo "   3. Review evaluation_report.txt for results"
