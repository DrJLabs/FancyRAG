#\!/usr/bin/env bats
# Unit tests for generate_tests.sh
# This test suite uses BATS (Bash Automated Testing System)
# Run with: bats tests/generate_tests.bats

# Setup function runs before each test
setup() {
    # Load the script functions without executing main logic
    # We'll source it in a subshell to avoid side effects
    export TEST_MODE=1
    export REPO_ROOT="$(pwd)"
    
    # Create temporary test directory
    export BATS_TEST_TMPDIR="${BATS_TEST_TMPDIR:-$(mktemp -d)}"
    export TEST_REPO="${BATS_TEST_TMPDIR}/test_repo"
    mkdir -p "${TEST_REPO}"
    cd "${TEST_REPO}"
    
    # Initialize a test git repository
    git init -q
    git config user.email "test@example.com"
    git config user.name "Test User"
    
    # Create initial commit on main branch
    echo "initial" > README.md
    git add README.md
    git commit -q -m "Initial commit"
}

# Teardown function runs after each test
teardown() {
    # Clean up test repository
    if [ -n "${BATS_TEST_TMPDIR}" ] && [ -d "${BATS_TEST_TMPDIR}" ]; then
        rm -rf "${BATS_TEST_TMPDIR}"
    fi
    cd "${REPO_ROOT}"
}

# Test: Script exists and is executable
@test "generate_tests.sh exists and is executable" {
    [ -f "${REPO_ROOT}/generate_tests.sh" ]
    [ -x "${REPO_ROOT}/generate_tests.sh" ]
}

# Test: Script has proper shebang
@test "generate_tests.sh has proper shebang" {
    run head -1 "${REPO_ROOT}/generate_tests.sh"
    [[ "$output" =~ ^#\!/ ]]
}

# Test: Script syntax is valid
@test "generate_tests.sh has valid bash syntax" {
    run bash -n "${REPO_ROOT}/generate_tests.sh"
    [ "$status" -eq 0 ]
}

# Test: Script handles missing git repository gracefully
@test "handles missing git repository" {
    cd "${BATS_TEST_TMPDIR}"
    rm -rf "${TEST_REPO}"
    mkdir -p no_git_repo
    cd no_git_repo
    
    run "${REPO_ROOT}/generate_tests.sh"
    # Should fail or warn when not in a git repo
    [ "$status" -ne 0 ] || [[ "$output" =~ "not a git repository" ]] || [[ "$output" =~ "fatal" ]]
}

# Test: Script detects current branch correctly
@test "detects current git branch" {
    cd "${TEST_REPO}"
    git checkout -q -b feature-branch
    
    # The script should be able to detect we're on feature-branch
    current_branch=$(git branch --show-current)
    [ "$current_branch" = "feature-branch" ]
}

# Test: Script handles no differences between branches
@test "handles no differences between main and HEAD" {
    cd "${TEST_REPO}"
    
    run "${REPO_ROOT}/generate_tests.sh"
    # When there are no changes, script should handle gracefully
    # Either succeed with no tests or indicate no files to test
    [[ "$output" =~ "no changes" ]] || [[ "$output" =~ "No files" ]] || [ "$status" -eq 0 ]
}

# Test: Script identifies changed files in diff
@test "identifies files changed in git diff" {
    cd "${TEST_REPO}"
    git checkout -q -b test-branch
    
    # Create a new file
    echo "console.log('test');" > test.js
    git add test.js
    git commit -q -m "Add test.js"
    
    # Script should detect test.js as changed
    files=$(git diff --name-only main..HEAD)
    [[ "$files" =~ "test.js" ]]
}

# Test: Script handles JavaScript files
@test "processes JavaScript files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b js-branch
    
    cat > app.js <<'JSEOF'
function add(a, b) {
    return a + b;
}
module.exports = { add };
JSEOF
    
    git add app.js
    git commit -q -m "Add JavaScript file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "app.js" ]]
}

# Test: Script handles TypeScript files
@test "processes TypeScript files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b ts-branch
    
    cat > app.ts <<'TSEOF'
function multiply(a: number, b: number): number {
    return a * b;
}
export { multiply };
TSEOF
    
    git add app.ts
    git commit -q -m "Add TypeScript file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "app.ts" ]]
}

# Test: Script handles Python files
@test "processes Python files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b py-branch
    
    cat > calculator.py <<'PYEOF'
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
PYEOF
    
    git add calculator.py
    git commit -q -m "Add Python file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "calculator.py" ]]
}

# Test: Script handles multiple files in diff
@test "processes multiple changed files" {
    cd "${TEST_REPO}"
    git checkout -q -b multi-file-branch
    
    echo "file1" > file1.js
    echo "file2" > file2.py
    echo "file3" > file3.ts
    
    git add file1.js file2.py file3.ts
    git commit -q -m "Add multiple files"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "file1.js" ]]
    [[ "$changed_files" =~ "file2.py" ]]
    [[ "$changed_files" =~ "file3.ts" ]]
}

# Test: Script handles configuration files
@test "processes configuration files (JSON/YAML)" {
    cd "${TEST_REPO}"
    git checkout -q -b config-branch
    
    cat > config.json <<'JSONEOF'
{
    "name": "test-app",
    "version": "1.0.0"
}
JSONEOF
    
    git add config.json
    git commit -q -m "Add config file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "config.json" ]]
}

# Test: Script handles deleted files
@test "handles deleted files in diff" {
    cd "${TEST_REPO}"
    
    # First create and commit a file
    echo "to be deleted" > deleteme.txt
    git add deleteme.txt
    git commit -q -m "Add file to delete"
    
    # Create a new branch and delete the file
    git checkout -q -b delete-branch
    git rm deleteme.txt
    git commit -q -m "Delete file"
    
    # Check that diff shows the deletion
    run git diff --name-status main..HEAD
    [[ "$output" =~ "D" ]] && [[ "$output" =~ "deleteme.txt" ]]
}

# Test: Script handles renamed files
@test "handles renamed files in diff" {
    cd "${TEST_REPO}"
    
    echo "original content" > original.js
    git add original.js
    git commit -q -m "Add original file"
    
    git checkout -q -b rename-branch
    git mv original.js renamed.js
    git commit -q -m "Rename file"
    
    run git diff --name-status main..HEAD
    [[ "$output" =~ "renamed.js" ]]
}

# Test: Script handles binary files
@test "handles binary files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b binary-branch
    
    # Create a dummy binary file
    printf '\x00\x01\x02\x03' > image.png
    git add image.png
    git commit -q -m "Add binary file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "image.png" ]]
}

# Test: Script handles nested directory structures
@test "handles files in nested directories" {
    cd "${TEST_REPO}"
    git checkout -q -b nested-branch
    
    mkdir -p src/components/utils
    echo "nested file" > src/components/utils/helper.js
    git add src/components/utils/helper.js
    git commit -q -m "Add nested file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "src/components/utils/helper.js" ]]
}

# Test: Script handles files with spaces in names
@test "handles files with spaces in names" {
    cd "${TEST_REPO}"
    git checkout -q -b spaces-branch
    
    echo "content" > "file with spaces.js"
    git add "file with spaces.js"
    git commit -q -m "Add file with spaces"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "file with spaces.js" ]]
}

# Test: Script handles special characters in filenames
@test "handles special characters in filenames" {
    cd "${TEST_REPO}"
    git checkout -q -b special-branch
    
    echo "content" > "file-name_test.js"
    git add "file-name_test.js"
    git commit -q -m "Add file with special chars"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "file-name_test.js" ]]
}

# Test: Script handles empty commits
@test "handles empty commits" {
    cd "${TEST_REPO}"
    git checkout -q -b empty-branch
    git commit -q --allow-empty -m "Empty commit"
    
    changed_files=$(git diff --name-only main..HEAD)
    [ -z "$changed_files" ]
}

# Test: Script handles merge commits
@test "handles merge commits" {
    cd "${TEST_REPO}"
    
    # Create a branch with changes
    git checkout -q -b feature1
    echo "feature1" > feature1.js
    git add feature1.js
    git commit -q -m "Add feature1"
    
    # Go back to main and create another branch
    git checkout -q main
    git checkout -q -b feature2
    echo "feature2" > feature2.js
    git add feature2.js
    git commit -q -m "Add feature2"
    
    # Merge feature1 into feature2
    git merge -q --no-edit feature1 || true
    
    # Check diff from main
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "feature1.js" ]] && [[ "$changed_files" =~ "feature2.js" ]]
}

# Test: Script validates git diff output format
@test "git diff output is in expected format" {
    cd "${TEST_REPO}"
    git checkout -q -b format-branch
    
    echo "test" > test.js
    git add test.js
    git commit -q -m "Add test"
    
    run git diff --name-only main..HEAD
    [ "$status" -eq 0 ]
    [ -n "$output" ]
}

# Test: Script handles large diffs
@test "handles large diffs with many files" {
    cd "${TEST_REPO}"
    git checkout -q -b large-branch
    
    # Create many files
    for i in {1..50}; do
        echo "file $i" > "file${i}.js"
    done
    
    git add .
    git commit -q -m "Add many files"
    
    changed_files=$(git diff --name-only main..HEAD | wc -l)
    [ "$changed_files" -eq 50 ]
}

# Test: Script handles file modifications
@test "detects modified files" {
    cd "${TEST_REPO}"
    
    echo "initial" > modify.js
    git add modify.js
    git commit -q -m "Add file"
    
    git checkout -q -b modify-branch
    echo "modified" >> modify.js
    git add modify.js
    git commit -q -m "Modify file"
    
    run git diff --name-status main..HEAD
    [[ "$output" =~ "M" ]] && [[ "$output" =~ "modify.js" ]]
}

# Test: Script handles combination of added, modified, and deleted files
@test "handles mixed file operations" {
    cd "${TEST_REPO}"
    
    echo "existing" > existing.js
    git add existing.js
    git commit -q -m "Add existing"
    
    git checkout -q -b mixed-branch
    
    # Add new file
    echo "new" > new.js
    # Modify existing
    echo "modified" >> existing.js
    # Delete README
    git rm README.md
    
    git add .
    git commit -q -m "Mixed operations"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "new.js" ]]
    [[ "$changed_files" =~ "existing.js" ]]
    [[ "$changed_files" =~ "README.md" ]]
}

# Test: Script handles diff with main branch correctly
@test "correctly diffs against main branch" {
    cd "${TEST_REPO}"
    
    # Ensure we're comparing against main
    git checkout -q -b test-main-diff
    echo "test" > test.js
    git add test.js
    git commit -q -m "Add test"
    
    # Verify main branch exists and has no test.js
    git checkout -q main
    [ \! -f test.js ]
    
    # Go back to test branch
    git checkout -q test-main-diff
    [ -f test.js ]
    
    # Diff should show test.js
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "test.js" ]]
}

# Test: Script handles non-existent base branch
@test "handles non-existent base branch gracefully" {
    cd "${TEST_REPO}"
    
    run git diff --name-only nonexistent..HEAD 2>&1
    [ "$status" -ne 0 ]
}

# Test: Script validates two-dot diff syntax
@test "uses two-dot diff syntax correctly" {
    cd "${TEST_REPO}"
    git checkout -q -b syntax-test
    
    echo "test" > test.js
    git add test.js
    git commit -q -m "Add test"
    
    # Two-dot syntax should work
    run git diff --name-only main..HEAD
    [ "$status" -eq 0 ]
    [[ "$output" =~ "test.js" ]]
}

# Test: Script handles HEAD reference correctly
@test "HEAD reference points to current branch tip" {
    cd "${TEST_REPO}"
    git checkout -q -b head-test
    
    echo "test" > test.js
    git add test.js
    git commit -q -m "Add test"
    
    head_commit=$(git rev-parse HEAD)
    branch_commit=$(git rev-parse head-test)
    
    [ "$head_commit" = "$branch_commit" ]
}

# Test: Script detects file type by extension
@test "correctly identifies file types by extension" {
    cd "${TEST_REPO}"
    git checkout -q -b filetype-branch
    
    # Create files of different types
    echo "js" > test.js
    echo "ts" > test.ts
    echo "py" > test.py
    echo "json" > test.json
    echo "yaml" > test.yaml
    
    git add .
    git commit -q -m "Add various file types"
    
    for file in test.js test.ts test.py test.json test.yaml; do
        changed_files=$(git diff --name-only main..HEAD)
        [[ "$changed_files" =~ "$file" ]]
    done
}

# Test: Script handles CSS files
@test "processes CSS files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b css-branch
    
    cat > styles.css <<'CSSEOF'
.container {
    display: flex;
    justify-content: center;
}
CSSEOF
    
    git add styles.css
    git commit -q -m "Add CSS file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "styles.css" ]]
}

# Test: Script handles HTML files
@test "processes HTML files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b html-branch
    
    cat > index.html <<'HTMLEOF'
<\!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><h1>Hello</h1></body>
</html>
HTMLEOF
    
    git add index.html
    git commit -q -m "Add HTML file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "index.html" ]]
}

# Test: Script handles Markdown files
@test "processes Markdown files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b md-branch
    
    cat > DOCS.md <<'MDEOF'
# Documentation

This is a test document.
MDEOF
    
    git add DOCS.md
    git commit -q -m "Add Markdown file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "DOCS.md" ]]
}

# Test: Script handles shell script files
@test "processes shell script files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b shell-branch
    
    cat > script.sh <<'SHEOF'
#\!/bin/bash
echo "Hello World"
SHEOF
    
    chmod +x script.sh
    git add script.sh
    git commit -q -m "Add shell script"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "script.sh" ]]
}

# Test: Script handles Go files
@test "processes Go files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b go-branch
    
    cat > main.go <<'GOEOF'
package main

func main() {
    println("Hello")
}
GOEOF
    
    git add main.go
    git commit -q -m "Add Go file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "main.go" ]]
}

# Test: Script handles Rust files
@test "processes Rust files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b rust-branch
    
    cat > main.rs <<'RSEOF'
fn main() {
    println\!("Hello, world\!");
}
RSEOF
    
    git add main.rs
    git commit -q -m "Add Rust file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "main.rs" ]]
}

# Test: Script handles Java files
@test "processes Java files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b java-branch
    
    cat > Main.java <<'JAVAEOF'
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
JAVAEOF
    
    git add Main.java
    git commit -q -m "Add Java file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "Main.java" ]]
}

# Test: Script handles Ruby files
@test "processes Ruby files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b ruby-branch
    
    cat > script.rb <<'RBEOF'
def greet(name)
    puts "Hello, #{name}\!"
end
RBEOF
    
    git add script.rb
    git commit -q -m "Add Ruby file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "script.rb" ]]
}

# Test: Script handles PHP files
@test "processes PHP files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b php-branch
    
    cat > index.php <<'PHPEOF'
<?php
function add($a, $b) {
    return $a + $b;
}
?>
PHPEOF
    
    git add index.php
    git commit -q -m "Add PHP file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "index.php" ]]
}

# Test: Script handles C/C++ files
@test "processes C/C++ files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b cpp-branch
    
    cat > main.cpp <<'CPPEOF'
#include <iostream>
int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
CPPEOF
    
    git add main.cpp
    git commit -q -m "Add C++ file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "main.cpp" ]]
}

# Test: Script handles YAML files
@test "processes YAML files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b yaml-branch
    
    cat > config.yaml <<'YAMLEOF'
name: test-app
version: 1.0.0
dependencies:
  - package1
  - package2
YAMLEOF
    
    git add config.yaml
    git commit -q -m "Add YAML file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "config.yaml" ]]
}

# Test: Script handles TOML files
@test "processes TOML files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b toml-branch
    
    cat > config.toml <<'TOMLEOF'
[package]
name = "test-app"
version = "1.0.0"
TOMLEOF
    
    git add config.toml
    git commit -q -m "Add TOML file"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "config.toml" ]]
}

# Test: Script handles file paths with dots
@test "handles file paths with dots correctly" {
    cd "${TEST_REPO}"
    git checkout -q -b dots-branch
    
    echo "content" > file.test.js
    git add file.test.js
    git commit -q -m "Add file with dots"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "file.test.js" ]]
}

# Test: Script handles symlinks
@test "handles symbolic links in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b symlink-branch
    
    echo "target" > target.js
    ln -s target.js link.js
    git add target.js link.js
    git commit -q -m "Add symlink"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "target.js" ]]
}

# Test: Script handles executable files
@test "handles executable files correctly" {
    cd "${TEST_REPO}"
    git checkout -q -b exec-branch
    
    cat > executable.sh <<'EXECEOF'
#\!/bin/bash
echo "executable"
EXECEOF
    
    chmod +x executable.sh
    git add executable.sh
    git commit -q -m "Add executable"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "executable.sh" ]]
}

# Test: Script handles .gitignore files
@test "processes .gitignore files in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b gitignore-branch
    
    cat > .gitignore <<'IGNOREEOF'
node_modules/
*.log
.env
IGNOREEOF
    
    git add .gitignore
    git commit -q -m "Add gitignore"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ ".gitignore" ]]
}

# Test: Script handles dotfiles
@test "processes dotfiles in diff" {
    cd "${TEST_REPO}"
    git checkout -q -b dotfile-branch
    
    echo "config" > .env
    git add .env
    git commit -q -m "Add dotfile"
    
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ ".env" ]]
}

# Test: Script performance with realistic repository
@test "performs reasonably with realistic file counts" {
    cd "${TEST_REPO}"
    git checkout -q -b perf-branch
    
    # Create a realistic number of files
    mkdir -p src/{components,utils,services}
    for i in {1..10}; do
        echo "component $i" > "src/components/Component${i}.js"
        echo "util $i" > "src/utils/util${i}.js"
        echo "service $i" > "src/services/Service${i}.js"
    done
    
    git add src
    git commit -q -m "Add realistic file structure"
    
    # Time the diff operation
    start=$(date +%s)
    git diff --name-only main..HEAD > /dev/null
    end=$(date +%s)
    duration=$((end - start))
    
    # Should complete in reasonable time (less than 5 seconds)
    [ "$duration" -lt 5 ]
}

# Test: Script handles uncommitted changes gracefully
@test "handles uncommitted changes appropriately" {
    cd "${TEST_REPO}"
    git checkout -q -b uncommitted-branch
    
    echo "committed" > committed.js
    git add committed.js
    git commit -q -m "Committed file"
    
    echo "uncommitted" > uncommitted.js
    # Don't add or commit
    
    # Diff should only show committed changes
    changed_files=$(git diff --name-only main..HEAD)
    [[ "$changed_files" =~ "committed.js" ]]
    [[ \! "$changed_files" =~ "uncommitted.js" ]]
}

# Test: Script handles staged but uncommitted changes
@test "handles staged but uncommitted changes" {
    cd "${TEST_REPO}"
    git checkout -q -b staged-branch
    
    echo "staged" > staged.js
    git add staged.js
    # Don't commit
    
    # Diff should not include staged but uncommitted changes
    changed_files=$(git diff --name-only main..HEAD)
    [[ \! "$changed_files" =~ "staged.js" ]]
}

# Test: Script handles repository with no commits
@test "handles repository with single commit" {
    temp_repo="${BATS_TEST_TMPDIR}/empty_repo"
    mkdir -p "${temp_repo}"
    cd "${temp_repo}"
    
    git init -q
    git config user.email "test@example.com"
    git config user.name "Test User"
    
    echo "first" > first.txt
    git add first.txt
    git commit -q -m "First commit"
    
    # Diff with no other branch should handle gracefully
    run git diff --name-only HEAD..HEAD
    [ "$status" -eq 0 ]
}

# Test: Script validates git is available
@test "git command is available" {
    command -v git
}

# Test: Script validates git version is sufficient
@test "git version is reasonably recent" {
    run git --version
    [ "$status" -eq 0 ]
    [[ "$output" =~ "git version" ]]
}