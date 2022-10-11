from aydin._version import version_tuple


def main():
    version_str = f"v{version_tuple[0]}.{version_tuple[1]}.{version_tuple[2] - 1}"

    content = f"""<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to master branch</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url={version_str}/index.html">
    <link rel="canonical" href="{version_str}/index.html">
  </head>
</html>"""

    with open("./build/html/index.html", "w") as index_file:
        # Writing data to a file
        index_file.write(content)


if __name__ == '__main__':
    main()
