def create_music_genre_hierarchy():
    return {
        'root': {
            'left': ['classical', 'opera'],
            'right': ['heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'jazz', 'blues', 'funk', 'country']
        },
        ('heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'jazz', 'blues', 'funk', 'country'): {
            'left': ['jazz', 'blues', 'funk', 'country'],
            'right': ['heavy-metal', 'rock', 'electronic', 'kids', 'pop']
        },
        ('classical', 'opera'): {
            'left': ['classical'],
            'right': ['opera']
        },
        ('jazz', 'blues', 'funk', 'country'): {
            'left': ['jazz', 'blues', 'funk'],
            'right': ['country']
        },
        ('jazz', 'blues', 'funk'): {
            'left': ['jazz'],
            'right': ['blues', 'funk']
        },
        ('blues', 'funk'): {
            'left': ['blues'],
            'right': ['funk']
        },
        ('heavy-metal', 'rock', 'electronic', 'kids', 'pop'): {
            'left': ['heavy-metal', 'rock', 'electronic'],
            'right': ['kids', 'pop']
        },
        ('heavy-metal', 'rock', 'electronic'): {
            'left': ['heavy-metal', 'rock'],
            'right': ['electronic']
        },
        ('heavy-metal', 'rock'): {
            'left': ['heavy-metal'],
            'right': ['rock']
        },
        ('kids', 'pop'): {
            'left': ['kids'],
            'right': ['pop']
        }
    }