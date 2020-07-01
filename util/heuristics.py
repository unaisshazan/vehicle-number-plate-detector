from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


is_in_the_same_row = lambda ya, yb: (abs(ya - yb) <= 20)
is_close_to = lambda x1a, x0b: (abs(x0b - x1a) <= 30)


def join_separated_2(bands):
    print('Bands size', len(bands))
    sorted_bands = sorted(bands, key=lambda tup: (tup[0], tup[2]))
    print("S bands", sorted_bands)
    new_bands = []

    while not sorted_bands == new_bands:
        # print("_____")
        # print("S bands", sorted_bands)
        # print("N bands", new_bands)

        skip = set()

        if new_bands:
            sorted_bands = new_bands
            new_bands = []

        for ida, left in enumerate(sorted_bands):
            for right in sorted_bands[ida:]:
                if should_join(left, right):
                    y0a, y1a, x0a, x1a = left
                    y0b, y1b, x0b, x1b = right

                    y0 = min(y0a, y0b)
                    y1 = max(y1a, y1b)
                    x0 = min(x0a, x0b)
                    x1 = max(x1a, x1b)
                    joined = (y0, y1, x0, x1)

                    if joined not in new_bands:
                        if left != right:
                            new_bands.append(joined)
                            skip.add(left)
                            skip.add(right)
                            if left in new_bands:
                                new_bands.remove(left)
                        elif left == sorted_bands[-1]:
                            if left not in skip:
                                new_bands.append(left)

                else:
                    if left not in new_bands:
                        if left not in skip:
                            new_bands.append(left)

    return sorted_bands


def should_join(left, right):
    y0a, y1a, x0a, x1a = left
    y0b, y1b, x0b, x1b = right
    return is_in_the_same_row(y0a, y0b) and is_in_the_same_row(y1a, y1b) and is_close_to(x1a, x0b)


def join_separated(bands):
    new_bands = set()

    bands = sorted(bands, key=lambda tup: (tup[0], tup[2]))
    # print('Bands', bands)

    skip_band = []
    for banda in bands:
        # print("Banda first", banda)
        if banda in skip_band:
            # print("Banda skip", banda, skip_band)
            continue
        y0a, y1a, x0a, x1a = banda
        one_row_bands = []

        for y0b, y1b, x0b, x1b in bands:
            if (is_in_the_same_row(y0a, y0b)) and (is_in_the_same_row(y1a, y1b)):
                one_row_bands.append((y0b, y1b, x0b, x1b))

        one_row_bands_new = []
        if len(one_row_bands) == 1:
            # print('Dodajemy bande 1', one_row_bands[0])
            new_bands.add(one_row_bands[0])

        for left, right in pairwise(one_row_bands):
            # print('Left', left, 'Right', right)
            if is_close_to(left[3], right[2]):
                one_row_bands_new.append(left)
                one_row_bands_new.append(right)
                skip_band.append(left)
                skip_band.append(right)
            else:
                # print('Dodajemy bande 2', left)
                if left not in skip_band:
                    new_bands.add(left)

        if len(one_row_bands_new) > 0:
            # print('One row bands new ', one_row_bands_new)
            y0 = min([band[0] for band in one_row_bands_new])
            y1 = max([band[1] for band in one_row_bands_new])
            x0 = min([band[2] for band in one_row_bands_new])
            x1 = max([band[3] for band in one_row_bands_new])
            # print('Dodajemy bande 3',(y0, y1, x0, x1))
            new_bands.add((y0, y1, x0, x1))

    # print('Last bands', new_bands)
    return list(set(new_bands))


def remove_big_areas(bands, size):
    area_picture = size[0] * size[1]

    bands_new = []
    for band in bands:
        y0, y1, x0, x1 = band
        area_box = ((y1 - y0) * (x1 - x0))

        prc = area_box / area_picture
        if prc <= 0.10:
            bands_new.append(band)
        else:
            print('Removed area', prc)

    return bands_new


def remove_vertical(bands, ratio_limit=0.6):
    bands_new = []
    for band in bands:
        y0, y1, x0, x1 = band

        ratio = (x1 - x0) / (y1 - y0)

        if ratio > ratio_limit:
            bands_new.append(band)
        else:
            print('Removed vertical ratio', ratio)

    return bands_new


def remove_horizontal(bands, width_image, percent_limit=0.4):
    bands_new = []
    for band in bands:
        y0, y1, x0, x1 = band
        width_box = (x1 - x0)

        prc = width_box / width_image
        if prc <= percent_limit:
            bands_new.append(band)
        else:
            print('Removed horizontal length', prc)

    return bands_new


def enhance_area(bands):
    bands_new = []
    up = 10
    for band in bands:
        y0, y1, x0, x1 = band
        y0 = y0 - up
        y1 = y1 + up
        x0 = x0 - up
        x1 = x1 + up

        bands_new.append((y0, y1, x0, x1))

    return bands_new

if __name__ == '__main__':
    bands = [(220, 252, 436, 487), (220, 252, 487, 976), (220, 252, 976, 1000), (255, 282, 42, 75)]

    # print(join_separated(bands))
    print(remove_big_areas(bands, (100,100)))
