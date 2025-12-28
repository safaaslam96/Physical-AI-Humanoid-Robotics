// Text highlighter plugin for Docusaurus
module.exports = function textHighlighterPlugin(context, options) {
  return {
    name: 'text-highlighter',
    injectHtmlTags() {
      return {
        postBodyTags: [
          '<script src="/js/text-highlighter.js"></script>'
        ],
      };
    },
  };
};
