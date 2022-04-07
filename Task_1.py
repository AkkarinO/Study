from scipy.stats import norm
from csv import writer


def generate_points_hor(num_points: int = 6000):
    placement_z = norm(loc=5, scale=50)
    placement_x = norm(loc=100, scale=50)
    placement_y = norm(loc=70, scale=0.05)

    x = placement_x.rvs(size=num_points)
    y = placement_y.rvs(size=num_points)
    z = placement_z.rvs(size=num_points)

    points_hor = zip(x, y, z)
    return points_hor


def generate_points_ver(num_points: int = 6000):
    placement_z = norm(loc=0, scale=50)
    placement_x = norm(loc=100, scale=0.05)
    placement_y = norm(loc=-90, scale=50)

    x = placement_x.rvs(size=num_points)
    y = placement_y.rvs(size=num_points)
    z = placement_z.rvs(size=num_points)

    points_ver = zip(x, y, z)
    return points_ver


def generate_points_cyl(num_points: int = 15000):
    placement_z = norm(loc=50, scale=30)
    placement_x = norm(loc=-120, scale=30)
    placement_y = norm(loc=20, scale=80)

    x = placement_x.rvs(size=num_points)
    y = placement_y.rvs(size=num_points)
    z = placement_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


if __name__ == '__main__':
    cloud_points_ver = generate_points_ver(6000)
    with open('LidarData_vertical.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points_ver:
            csvwriter.writerow(p)

    cloud_points_hor = generate_points_hor(6000)
    with open('LidarData_horizontal.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points_hor:
            csvwriter.writerow(p)

    cloud_points_cyl = generate_points_cyl(15000)
    with open('LidarData_cylindrical.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_points_cyl:
            csvwriter.writerow(p)

    # Joining three generated LidarData files into one 'LidarData_all'
    filenames = ['LidarData_vertical.xyz', 'LidarData_horizontal.xyz', 'LidarData_cylindrical.xyz']
    with open('LidarData_all.xyz', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
