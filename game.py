# Tic-Tac-Toe - Terminal Game

board = [" " for _ in range(9)]


def print_board():
    print()
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---+---+---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---+---+---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print()


def check_winner(player):
    combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in combinations:
        if board[a] == board[b] == board[c] == player:
            return True

    return False


def game():
    player = "X"

    while True:
        print_board()

        try:
            position = int(input(f"Player {player}, choose a position (1-9): "))
        except ValueError:
            print("Please enter a number!")
            continue

        if position < 1 or position > 9:
            print("Choose a number from 1 to 9.")
            continue

        position -= 1

        if board[position] != " ":
            print("That position is already taken!")
            continue

        board[position] = player

        if check_winner(player):
            print_board()
            print(f"🎉 Player {player} wins!")
            break

        if " " not in board:
            print_board()
            print("🤝 It's a draw!")
            break

        player = "O" if player == "X" else "X"


game()