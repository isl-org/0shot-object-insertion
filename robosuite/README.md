## Visualizing a frame
- Add a site in the body e.g.
```xml
<body>
  <...>
  <site name="vis_site" group="2"/>
</body>
```
- Add the following lines in `_reset_internal()` of [`base.py`](robosuite/environments/base.py):
```python
# under if self.has_renderer and self.viewer is None:
self.viewer.viewer.vopt.frame = 3  # vis sites only
self.viewer.viewer.vopt.sitegroup[:] = 0  # no site group, except...
self.viewer.viewer.vopt.sitegroup[2] = 1  # site group 2
```

[Original README](README_original.md)

Contributors:
- Original `robosuite` contributors
- Samarth Brahmbhatt
- Ankur Deka
- Andrew Spielberg
