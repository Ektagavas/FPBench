import imgkit

data = {
    "id": "N/A",
    "name_left": "101_2",
    "dob_left": "N/A",
    "case_id_left": "N/A",
    "name_right": "101_7",
    "dob_right": "N/A",
    "case_id_right": "N/A",
    "level1_left": "Finger: Distal Phalange, Right Loop",
    "level2_left": "Sufficient",
    "level1_right": "Finger: Distal Phalange, Right Loop",
    "level2_right": "Sufficient",
    "level3_left": "YES",
    "level3_right": "NO",
    "quality_left": "MEDIUM HIGH",
    "quality_right": "MEDIUM HIGH",
    "suitable_left": True,
    "suitable_right": True,
    "fingerprint_left": True,
    "fingerprint_right": True,
    "level1_agreement": True,
    "level2_agreement": True,
    "evaluation": "EXCLUSION"
}

html = f"""
<html>
<head>
  <style>
    body {{
      font-family: Arial, sans-serif;
      font-size: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    td, th {{
      border: 1px solid black;
      padding: 6px;
      vertical-align: middle;
    }}
    .section-title {{
      font-weight: bold;
      background-color: #f2f2f2;
    }}
    .checkbox {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 1px solid #000;
      margin-right: 3px;
      vertical-align: middle;
    }}
    .checked::after {{
  content: "\\2713";
  color: black;
  font-size: 10px;
  position: relative;
  left: 1px;
  top: -2px;
}}

  </style>
</head>
<body>

<table>
  <tr><td><b>ID #</b></td><td>{data["id"]}</td></tr>
  <tr><td colspan="2">Analysis must be consistent with SWGFAST terminology and definitions, and shall include, but not be limited to, the following elements:</td></tr>
</table>

<h3>ANALYSIS</h3>
<table>
  <tr>
    <td>Name: {data["name_left"]}</td><td>DOB: {data["dob_left"]}</td><td>Case ID: {data["case_id_left"]}</td>
    <td>Name: {data["name_right"]}</td><td>DOB: {data["dob_right"]}</td><td>Case ID: {data["case_id_right"]}</td>
  </tr>

  <tr>
    <td colspan="3">Level 1: {data["level1_left"]}<br>Level 2: {data["level2_left"]}</td>
    <td colspan="3">Level 1: {data["level1_right"]}<br>Level 2: {data["level2_right"]}</td>
  </tr>

  <tr>
    <td colspan="3">Level 3 Visible? {data["level3_left"]}</td>
    <td colspan="3">Level 3 Visible? {data["level3_right"]}</td>
  </tr>

  <tr>
    <td>QUALITY</td>
    <td colspan="2">{data["quality_left"]}</td>
    <td>QUALITY</td>
    <td colspan="2">{data["quality_right"]}</td>
  </tr>

  <tr>
    <td>SUITABLE FOR COMPARISON</td>
    <td colspan="2"><span class="checkbox {'checked' if data['suitable_left'] else ''}"></span></td>
    <td>SUITABLE FOR COMPARISON</td>
    <td colspan="2"><span class="checkbox {'checked' if data['suitable_right'] else ''}"></span></td>
  </tr>

  <tr>
    <td colspan="2">Fingerprint <span class="checkbox {'checked' if data['fingerprint_left'] else ''}"></span></td>
    <td colspan="2">Palm print <span class="checkbox"></span></td>
    <td colspan="2">Footprint <span class="checkbox"></span></td>
  </tr>
</table>

<h3>COMPARISON</h3>
<table>
  <tr>
    <td colspan="2">LEVEL 1 AGREEMENT <span class="checkbox {'checked' if data['level1_agreement'] else ''}"></span></td>
    <td colspan="2">LEVEL 2 AGREEMENT <span class="checkbox {'checked' if data['level2_agreement'] else ''}"></span></td>
    <td colspan="2">LEVEL 3 AGREEMENT <span class="checkbox"></span></td>
  </tr>
</table>

<h3>EVALUATION</h3>
<table>
  <tr>
    <td>INDIVIDUALIZATION</td>
    <td>EXCLUSION <span class="checkbox {'checked' if data['evaluation'] == 'EXCLUSION' else ''}"></span></td>
    <td>INCONCLUSIVE <span class="checkbox {'checked' if data['evaluation'] == 'INCONCLUSIVE' else ''}"></span></td>
  </tr>
</table>

</body>
</html>
"""

# save HTML and convert
with open("acev_sheet.html", "w") as f:
    f.write(html)

imgkit.from_file("acev_sheet.html", "acev_sheet.png")
