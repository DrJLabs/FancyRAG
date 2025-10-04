# Tests for generate_tests.sh

This directory contains comprehensive unit tests for the `generate_tests.sh` script.

## Prerequisites

The tests use BATS (Bash Automated Testing System). Install it with:

### On Ubuntu/Debian
```bash
sudo apt-get install bats
```

### On macOS
```bash
brew install bats-core
```

### Manual Installation
```bash
git clone https://github.com/bats-core/bats-core.git
cd bats-core
sudo ./install.sh /usr/local
```

## Running Tests

Run all tests:
```bash
bats tests/generate_tests.bats
```

Run with verbose output:
```bash
bats -t tests/generate_tests.bats
```

Run specific test:
```bash
bats -f "test name pattern" tests/generate_tests.bats
```

## Test Coverage

The test suite covers:

1. **Basic Validation**
   - Script existence and executability
   - Syntax validation
   - Shebang verification

2. **Git Operations**
   - Repository detection
   - Branch detection
   - Diff operations (two-dot syntax)
   - Handling various git states

3. **File Type Support**
   - JavaScript (.js)
   - TypeScript (.ts)
   - Python (.py)
   - Go (.go)
   - Rust (.rs)
   - Java (.java)
   - Ruby (.rb)
   - PHP (.php)
   - C/C++ (.c, .cpp)
   - Shell scripts (.sh)
   - Configuration files (JSON, YAML, TOML)
   - Markup (HTML, Markdown, CSS)
   - Dotfiles

4. **Edge Cases**
   - Empty commits
   - Multiple file operations
   - Deleted files
   - Renamed files
   - Binary files
   - Files with special characters
   - Nested directories
   - Symlinks
   - Large diffs

5. **Error Handling**
   - Missing git repository
   - Non-existent branches
   - Uncommitted changes
   - Staged but uncommitted changes

6. **Performance**
   - Reasonable execution time with many files

## Contributing

When adding new tests:
1. Follow BATS conventions
2. Use descriptive test names
3. Include setup/teardown as needed
4. Test both success and failure cases
5. Add comments for complex test logic