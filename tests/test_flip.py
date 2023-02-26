from utils.flip import convert_ch, flip_fen, flip_uci


def test_convert_ch():
    assert convert_ch('1') == '1'
    assert convert_ch('-') == '-'
    assert convert_ch('K') == 'k'
    assert convert_ch('k') == 'K'

    for piece in 'prnbqkPRNBQK':
        assert piece == convert_ch(convert_ch(piece))


def test_flip_fen():
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0'
    assert fen == flip_fen(flip_fen(fen))

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 0'
    assert fen == flip_fen(flip_fen(fen))

    fen = 'rr1n4/bbppnqpp/6k1/p3ppP1/1pPP3P/1N2PN1B/PPQBKP1R/2R5 b - - 0 0'
    assert fen == flip_fen(flip_fen(fen))

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq b6 0 0'
    assert fen == flip_fen(flip_fen(fen))

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq c3 0 0'
    assert fen == flip_fen(flip_fen(fen))

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 0'
    assert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 0' == flip_fen(fen)

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 0'
    assert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w kq - 0 0' == flip_fen(fen)

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kk - 0 0'
    assert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kk - 0 0' == flip_fen(fen)

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w kq - 0 0'
    assert 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 0' == flip_fen(fen)


def test_flip_uci():
    uci = 'e2e4'
    assert uci == flip_uci(flip_uci(uci))

    uci = 'e7e5'
    assert uci == flip_uci(flip_uci(uci))

    uci = 'e2e4'
    assert 'e7e5' == flip_uci(uci)

    uci = 'e7e5'
    assert 'e2e4' == flip_uci(uci)

    uci = 'b1c3'
    assert 'b8c6' == flip_uci(uci)

    uci = 'b8c6'
    assert 'b1c3' == flip_uci(uci)

    uci = 'e1g1'
    assert 'e8g8' == flip_uci(uci)

    uci = 'd7d8q'
    assert 'd2d1q' == flip_uci(uci)
    assert 'd7d8q' == flip_uci(flip_uci(uci))

    uci = 'b2b1k'
    assert 'b7b8k' == flip_uci(uci)
    assert 'b2b1k' == flip_uci(flip_uci(uci))
