### rdf_calculator_ovito.py

---

This script will read the `dump.xyz` file and then calculate the RDF by using the `ovito` package.

#### Usage

```
python rdf_calculator_ovito.py <extxyz_file> <cutoff> <bins>
```

- `<extxyz_file>`: The path to the `extxyz` file.
- `<cutoff>`: The cutoff used in the RDF calculation.
- `<bins>`: The bins used in the RDF calculation.

#### Example

```sh
python rdf_calculator_ovito.py dump.xyz 6 400
```

This command will read the `dump.xyz` file and calculate the RDF by using the `ovito` package and output the `rdf.txt` file.



---

Thank you for using `GPUMDkit`! If you have any questions or need further assistance, feel free to open an issue on our GitHub repository or contact Zihan YAN (yanzihan@westlake.edu.cn).
