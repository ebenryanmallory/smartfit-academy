// Python to JavaScript code conversion utility

/**
 * Converts Python code to JavaScript with basic syntax transformations
 * @param pythonCode - The Python code to convert
 * @returns The converted JavaScript code
 */
export function convertPythonToJS(pythonCode: string): string {
  // Basic conversions
  let jsCode = pythonCode
    .replace(/print\((.*?)\)/g, 'console.log($1)') // print() -> console.log()
    .replace(/def /g, 'function ') // def -> function
    .replace(/:$/gm, ' {') // : -> {
    .replace(/^(\s*)(?=\S)/gm, '$1  ') // Indentation
    .replace(/\n/g, '\n  ') // Add indentation
    .replace(/\n\s*\n/g, '\n}\n\n') // Add closing braces
    .replace(/True/g, 'true') // True -> true
    .replace(/False/g, 'false') // False -> false
    .replace(/None/g, 'null') // None -> null
    .replace(/f"(.*?)"/g, '`$1`') // f-strings -> template literals
    .replace(/\{([^}]+)\}/g, '${$1}') // {var} -> ${var}
    .replace(/#/g, '//'); // Comments

  // Add closing brace for the last function
  if (jsCode.includes('function ')) {
    jsCode += '\n}';
  }

  return jsCode;
}