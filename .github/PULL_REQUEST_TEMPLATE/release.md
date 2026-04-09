# Release: vX.Y.Z

## Summary

This pull request prepares the **vX.Y.Z release** of athena by updating the `main` branch with the latest stable changes from `development`.

The release includes bug fixes, improvements, and new features stabilised during the development cycle.

---

## Release Checklist

### Versioning

* [ ] Version updated to `X.Y.Z`
* [ ] `CHANGELOG` updated
* [ ] Release date added to changelog
* [ ] Version string updated in source files (if applicable)

### Documentation

* [ ] Documentation builds successfully
* [ ] New features documented
* [ ] API changes documented

### Testing

* [ ] All unit tests pass
* [ ] Example scripts run successfully
* [ ] No regressions in existing functionality

### Build

* [ ] `fpm build` succeeds
* [ ] `fpm test` succeeds
* [ ] `fpm run --example` succeeds

---

## Changes in this Release

### Added

*

### Changed

*

### Fixed

*

### Removed

*

---

## Release Procedure

1. Rebase `development` onto `main`
2. Update version numbers and changelog
3. Merge this PR into `main`
4. Create release tag:

```
git tag vX.Y.Z
git push origin vX.Y.Z
```

5. Verify GitHub release archive
6. Update external package managers (e.g. Spack)

---

## Notes for Reviewers

* Confirm version bump is correct
* Verify changelog accuracy
* Ensure CI passes on `main`

---

## Post-Merge Tasks

* [ ] Create GitHub release
* [ ] Announce release if appropriate
* [ ] Update package manager recipes
